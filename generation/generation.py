import argparse
import json
import os
import pickle
import re
import sys

import pandas as pd
import torch
from accelerate import Accelerator
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher as SimpleSearcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation import exact_match_multiple, normalize_answer
from utils.prompt_template import (
    SELECTOR_SYSTEM_PROMPT,
    SUFF_SYSTEM_PROMPT,
    TRIVIAQA_FORCE_ANSWER_PROMPT,
    TRIVIAQA_SELECTOR_SYSTEM_PROMPT,
    force_answer_prompt,
)
from utils.text_processing import (
    extract_evidence_global_ids,
    extract_final_answer_and_rationale,
    extract_gap_items,
    safe_literal_eval,
)
from utils.corpus_selection import get_bm25_resource_paths, infer_dataset_name

try:
    import pysbd

    _PYSBD_OK = True
    _PYSBD_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
except Exception:
    pysbd = None
    _PYSBD_OK = False
    _PYSBD_SEGMENTER = None

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_JSON_RE = re.compile(r"\{.*\}", re.S)

CSV_COLUMNS = [
    "ID",
    "Turn",
    "Reasoner Task Content",
    "Evidence Context So Far",
    "Raw Retrieved Docs",
    "Gate Output",
    "SuffMissingFacts",
    "Reasoner Answer",
    "Gold Answer",
    "Correct Answer",
    "Gold Retrieved Docs",
    "Retrieved Titles So Far",
    "Correct Retrieval",
]

accelerator = None
tokenizer = None
model = None
gate_tokenizer = None
gate_model = None
searcher = None
corpus = None
MODEL = None


def get_selector_system_prompt(dataset_name):
    """Choose a selector prompt that matches the dataset's evidence style."""
    if str(dataset_name or "").strip().lower() == "triviaqa":
        return TRIVIAQA_SELECTOR_SYSTEM_PROMPT
    return SELECTOR_SYSTEM_PROMPT


def get_answer_system_prompt(dataset_name):
    """Choose an answer-generation prompt that matches the dataset's answer style."""
    if str(dataset_name or "").strip().lower() == "triviaqa":
        return TRIVIAQA_FORCE_ANSWER_PROMPT
    return force_answer_prompt


def safe_json_load(text: str):
    """Parse JSON robustly, including fenced code blocks."""
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def split_wiki_sentences(text: str):
    """Split Wikipedia-like text into sentences."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n+", text)
    sentences = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        cjk_count = len(_CJK_RE.findall(paragraph))
        if cjk_count > 0 and (cjk_count / max(1, len(paragraph))) > 0.02:
            parts = re.split(r"(?<=[。！？!?])", paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())
            continue

        if _PYSBD_OK:
            parts = _PYSBD_SEGMENTER.segment(paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())
        else:
            parts = re.split(r"(?<=[.!?])\s+|[\n]+", paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())

    return sentences


def concat_raw_retrieved_docs(titles, texts):
    """Concatenate raw retrieved documents before sentence selection."""
    blocks = []
    for title, raw_text in zip(titles or [], texts or []):
        if not raw_text:
            continue
        blocks.append(f"[{title or 'N/A'}]\n{raw_text.strip()}")
    return "\n\n---\n\n".join(blocks)


def call_reasoner_batch(
    system_prefixes,
    task_contents,
    gpt=False,
    temperature=0.6,
    top_p=0.9,
    max_new_tokens=256,
    do_sample=True,
):
    """Run the main reasoner model and return decoded outputs."""
    if isinstance(system_prefixes, str):
        messages = [
            [
                {"role": "system", "content": system_prefixes},
                {"role": "user", "content": task_content},
            ]
            for task_content in task_contents
        ]
    else:
        messages = [
            [
                {"role": "system", "content": system_prefix},
                {"role": "user", "content": task_content},
            ]
            for task_content, system_prefix in zip(task_contents, system_prefixes)
        ]

    if gpt:
        api_key = os.environ.get("CHATANYWHERE_API_KEY")
        if not api_key:
            raise RuntimeError("Please set CHATANYWHERE_API_KEY.")

        client = OpenAI(
            base_url="https://api.chatanywhere.tech/v1",
            api_key=api_key,
        )

        outputs = []
        for message in messages:
            response = client.chat.completions.create(
                model=MODEL,
                messages=message,
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=None,
            )
            outputs.append(response.choices[0].message.content)
        return outputs

    prompts = [
        tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        for msgs in messages
    ]
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(accelerator.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [text.strip() for text in decoded]


def bm25_search_batch(query_texts, all_past_doc_ids, k=2, remove_repeat_docs=True):
    """Run BM25 search with optional exact-doc de-duplication."""
    past_doc_id_sets = [set(x) for x in all_past_doc_ids] if remove_repeat_docs else [set() for _ in all_past_doc_ids]

    titles_batch = []
    texts_batch = []
    doc_ids_batch = []

    for query_index, query_text in enumerate(query_texts):
        hits = searcher.search(str(query_text or "").strip(), k=max(k, 50))
        filtered_doc_ids = []
        for hit in hits:
            doc_id = hit.docid
            if doc_id not in corpus:
                continue
            if remove_repeat_docs and doc_id in past_doc_id_sets[query_index]:
                continue
            filtered_doc_ids.append(doc_id)
            if len(filtered_doc_ids) >= k:
                break

        top_titles = []
        top_texts = []
        top_doc_ids = []
        for doc_id in filtered_doc_ids:
            top_doc_ids.append(doc_id)
            top_titles.append(corpus[doc_id].get("title", "No results found."))
            top_texts.append(corpus[doc_id].get("text", "No results found."))

        if not top_titles:
            top_doc_ids = [""]
            top_titles = ["No results found."]
            top_texts = ["No results found."]

        titles_batch.append(top_titles)
        texts_batch.append(top_texts)
        doc_ids_batch.append(top_doc_ids)

    return titles_batch, texts_batch, doc_ids_batch


def build_suff_user_prompt(question: str, context: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context}

Please ONLY use the above CONTEXT as evidence.
Decide whether it is sufficient, and if not, list the missing facts
in the required JSON format."""


def call_suff_gate_batch(questions, evidence_contexts, gate_model_obj, gate_tokenizer_obj, max_new_tokens=256):
    """Run the sufficiency gate using only accumulated evidence context, and return sufficiency flags plus missing facts."""
    messages_batch = []
    for question, context in zip(questions, evidence_contexts):
        messages_batch.append(
            [
                {"role": "system", "content": SUFF_SYSTEM_PROMPT},
                {"role": "user", "content": build_suff_user_prompt(question, context)},
            ]
        )

    prompts = [
        gate_tokenizer_obj.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        for msgs in messages_batch
    ]
    inputs = gate_tokenizer_obj(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(accelerator.device)

    with torch.no_grad():
        outputs = gate_model_obj.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_len:]
    texts = gate_tokenizer_obj.batch_decode(generated_ids, skip_special_tokens=True)

    verdicts = []
    missing_facts_batch = []

    for assistant_text in texts:
        parsed = safe_json_load((assistant_text or "").strip())
        if not isinstance(parsed, dict):
            verdicts.append(False)
            missing_facts_batch.append([])
            continue

        sufficient = bool(parsed.get("sufficient", False))
        missing_facts = extract_gap_items(parsed)
        if sufficient or not isinstance(missing_facts, list):
            missing_facts = []

        verdicts.append(sufficient)
        missing_facts_batch.append(missing_facts)

    return verdicts, missing_facts_batch


def build_query_from_missing(question, missing_facts, max_facts=None, dataset_name=None):
    """Build a retrieval query from missing facts."""
    if max_facts is None:
        max_facts = 1 if str(dataset_name or "").strip().lower() == "triviaqa" else 3

    if not missing_facts:
        return question

    phrases = []
    for item in missing_facts[:max_facts]:
        if not isinstance(item, dict):
            continue

        target = (item.get("target") or "").strip()
        slot = (item.get("slot") or "").strip()
        description = (item.get("description") or "").strip()

        if target and slot:
            phrases.append(f"{target} {slot}")
        elif description:
            phrases.append(description)

    return question if not phrases else question + " " + " ".join(phrases)


def should_force_first_retrieval(turn, evidence_context, verdict):
    """Force one retrieval when the first-turn gate is overconfident on empty evidence."""
    return turn == 0 and bool(verdict) and not str(evidence_context or "").strip()


def format_missing_facts_for_selector(missing_facts, max_facts=3) -> str:
    """Format missing-fact hints for the selector prompt."""
    if not missing_facts:
        return "None."

    lines = []
    for item in missing_facts[:max_facts]:
        if not isinstance(item, dict):
            continue

        category = (item.get("category") or "").strip()
        target = (item.get("target") or "").strip()
        slot = (item.get("slot") or "").strip()
        description = (item.get("description") or "").strip()

        core = f"{target} {slot}".strip() if target and slot else description
        if core:
            lines.append(f"- [{category or 'other'}] {core}")

    return "\n".join(lines) if lines else "None."


def concat_and_pick_sentences_batch(
    questions,
    titles_batch,
    texts_batch,
    missing_facts_batch=None,
    dataset_name=None,
    gpt=False,
    return_top_k=6,
    max_sents_per_doc=40,
):
    """Select sentence-level evidence from retrieved documents."""
    tasks = []
    index_maps = []

    if missing_facts_batch is None:
        missing_facts_batch = [[] for _ in questions]

    for i, (titles, texts) in enumerate(zip(titles_batch, texts_batch)):
        question = questions[i]
        missing_facts = missing_facts_batch[i]

        sentences = []
        for doc_idx, (title, raw_text) in enumerate(zip(titles, texts)):
            if not raw_text or raw_text.strip() == "No results found.":
                continue

            parts = split_wiki_sentences(raw_text)
            if max_sents_per_doc is not None:
                parts = parts[:max_sents_per_doc]

            for local_sid, sentence_text in enumerate(parts, start=1):
                sentences.append((doc_idx, local_sid, title or "N/A", sentence_text))

        if not sentences:
            tasks.append(None)
            index_maps.append({})
            continue

        numbered_lines = []
        id_map = {}
        for global_id, (doc_idx, local_sid, title, sentence_text) in enumerate(sentences, start=1):
            numbered_lines.append(f"[{global_id}] ({title} | s#{local_sid}) {sentence_text}")
            id_map[global_id] = (doc_idx, local_sid)

        hints = format_missing_facts_for_selector(missing_facts, max_facts=3)
        user_prompt = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"MISSING FACTS TO FILL:\n{hints}\n\n"
            f"NUMBERED SENTENCES FROM RETRIEVED DOCUMENTS:\n"
            + "\n".join(numbered_lines)
            + "\n\n"
            + f"You may select up to {return_top_k} sentences.\n"
            + 'Return ONLY JSON with "evidence_global_ids".'
        )

        tasks.append(user_prompt)
        index_maps.append(id_map)

    tasks_to_call = [task for task in tasks if task is not None]
    outputs = []
    if tasks_to_call:
        outputs = call_reasoner_batch(
            system_prefixes=get_selector_system_prompt(dataset_name),
            task_contents=tasks_to_call,
            gpt=gpt,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=64,
            do_sample=False,
        )

    picked_ids_batch = []
    output_ptr = 0

    for i, titles in enumerate(titles_batch):
        if tasks[i] is None:
            picked_ids_batch.append([[] for _ in titles])
            continue

        raw_output = (outputs[output_ptr] or "").strip()
        output_ptr += 1

        parsed = safe_json_load(raw_output)
        if not isinstance(parsed, dict):
            parsed = {}

        global_ids = []
        values = extract_evidence_global_ids(parsed)
        if isinstance(values, list):
            for value in values:
                try:
                    value = int(value)
                    if value >= 1:
                        global_ids.append(value)
                except Exception:
                    pass

        per_doc = [[] for _ in titles]
        for global_id in global_ids:
            if global_id not in index_maps[i]:
                continue
            doc_idx, local_sid = index_maps[i][global_id]
            if 0 <= doc_idx < len(per_doc):
                per_doc[doc_idx].append(int(local_sid))

        picked_ids_batch.append([sorted(set(x)) for x in per_doc])

    return picked_ids_batch


def merge_evidence_only(titles, texts, evidence_ids_per_doc):
    """Keep only selected evidence sentences in a merged block."""
    blocks = []
    for title, raw_text, sentence_ids in zip(titles, texts, evidence_ids_per_doc):
        if not raw_text or not sentence_ids:
            continue

        sentences = split_wiki_sentences(raw_text)
        chosen = []
        for index in sentence_ids:
            if 1 <= index <= len(sentences):
                chosen.append(sentences[index - 1])

        if chosen:
            blocks.append(f"[{title or 'N/A'}] EVIDENCE:\n- " + "\n- ".join(chosen))

    return "\n\n---\n\n".join(blocks)


def append_evidence_context(previous: str, new_text: str, sep: str = "\n\n---\n\n") -> str:
    """Append new evidence text while avoiding exact duplication."""
    previous = (previous or "").strip()
    new_text = (new_text or "").strip()

    if not new_text:
        return previous
    if not previous:
        return new_text
    if new_text in previous:
        return previous
    return previous + sep + new_text


def update_task_with_evidence(task_content: str, query: str, evidence_block: str) -> str:
    """Append retrieved evidence to the task content."""
    evidence_block = (evidence_block or "").strip()
    if not evidence_block:
        return task_content
    return f"{task_content}Query: {query}\nRetrieved Document: {evidence_block}\n"


def save_results_to_csv(df: pd.DataFrame, filename: str):
    """Append results to CSV with a stable column order."""
    df = df.reindex(columns=CSV_COLUMNS)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="a", header=False, index=False)


def _load_gold_answers(row):
    """Support both 'Answer' and legacy 'Answers' columns."""
    if "Answers" in row.index:
        return safe_literal_eval(row["Answers"])
    if "Answer" in row.index:
        value = row["Answer"]
        parsed = safe_literal_eval(value)
        return parsed if isinstance(parsed, list) else [value]
    return []


def _load_gold_documents(row):
    """Load gold document titles from the CSV row."""
    if "Documents" not in row.index:
        return []
    return safe_literal_eval(row["Documents"])


def main_batch(
    task_contents,
    question_type,
    ids,
    batch_gold_answers,
    gold_retrieved_docs,
    checkpoint_file,
    max_turns,
    dataset_name,
    gpt,
    top_docs,
    remove_repeat_docs,
    batch_questions=None,
    write_csv=True,
):
    """Run one batched multi-turn RAG loop until max_turns without early stopping."""
    if batch_questions is None:
        batch_questions = [""] * len(task_contents)

    total_turns = int(max_turns or 4)
    results = []
    all_past_titles = [[] for _ in task_contents]
    all_past_doc_ids = [[] for _ in task_contents]
    evidence_context_so_far = [""] * len(task_contents)
    raw_retrieved_concat_so_far = [""] * len(task_contents)

    for turn in range(total_turns):
        verdicts, missing_facts_list = call_suff_gate_batch(
            batch_questions,
            evidence_context_so_far,
            gate_model,
            gate_tokenizer,
        )

        for i, verdict in enumerate(verdicts):
            if should_force_first_retrieval(turn, evidence_context_so_far[i], verdict):
                verdicts[i] = False
                missing_facts_list[i] = []

        sub_queries = [
            build_query_from_missing(
                batch_questions[i],
                missing_facts_list[i],
                dataset_name=dataset_name,
            )
            for i in range(len(task_contents))
        ]
        sub_past_doc_ids = all_past_doc_ids

        sub_titles, sub_texts, sub_doc_ids = bm25_search_batch(
            sub_queries,
            sub_past_doc_ids,
            k=top_docs,
            remove_repeat_docs=remove_repeat_docs,
        )

        for i in range(len(task_contents)):
            for doc_id, title in zip(sub_doc_ids[i], sub_titles[i]):
                if doc_id and doc_id != "No results found." and doc_id not in all_past_doc_ids[i]:
                    all_past_doc_ids[i].append(doc_id)
                if title and title != "No results found." and title not in all_past_titles[i]:
                    all_past_titles[i].append(title)

            new_raw = concat_raw_retrieved_docs(sub_titles[i], sub_texts[i])
            if not new_raw:
                continue
            if raw_retrieved_concat_so_far[i]:
                raw_retrieved_concat_so_far[i] += "\n\n---\n\n" + new_raw
            else:
                raw_retrieved_concat_so_far[i] = new_raw

        sub_evidence_ids = concat_and_pick_sentences_batch(
            questions=batch_questions,
            titles_batch=sub_titles,
            texts_batch=sub_texts,
            missing_facts_batch=missing_facts_list,
            dataset_name=dataset_name,
            gpt=gpt,
            return_top_k=6,
            max_sents_per_doc=40,
        )

        for i in range(len(task_contents)):
            merged = merge_evidence_only(
                sub_titles[i],
                sub_texts[i],
                sub_evidence_ids[i],
            )
            evidence_context_so_far[i] = append_evidence_context(
                evidence_context_so_far[i],
                merged,
            )
            task_contents[i] = update_task_with_evidence(
                task_contents[i],
                sub_queries[i],
                merged,
            )

        predicted_answers = {}
        correct_answer_flags = {}

        if turn == total_turns - 1:
            done_outputs = call_reasoner_batch(
                system_prefixes=get_answer_system_prompt(dataset_name),
                task_contents=task_contents,
                gpt=gpt,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=128,
                do_sample=False,
            )

            for global_index in range(len(task_contents)):
                pred_answer, _ = extract_final_answer_and_rationale(
                    done_outputs[global_index],
                    question_type,
                )
                pred_answer_norm = normalize_answer(pred_answer)
                predicted_answers[global_index] = pred_answer_norm
                correct_answer_flags[global_index] = bool(
                    exact_match_multiple(pred_answer_norm, batch_gold_answers[global_index])
                )

        for i in range(len(task_contents)):
            unique_titles = list(dict.fromkeys(all_past_titles[i]))
            correct_retrieval = all(
                exact_match_multiple(gold_title, unique_titles)
                for gold_title in gold_retrieved_docs[i]
            )

            pred_answer_norm = predicted_answers.get(i, "")
            correct_answer = correct_answer_flags.get(i, False) if pred_answer_norm else False

            results.append(
                {
                    "ID": ids[i],
                    "Turn": turn + 1,
                    "Reasoner Task Content": task_contents[i],
                    "Evidence Context So Far": evidence_context_so_far[i],
                    "Raw Retrieved Docs": raw_retrieved_concat_so_far[i],
                    "Gate Output": bool(verdicts[i]),
                    "SuffMissingFacts": missing_facts_list[i],
                    "Reasoner Answer": pred_answer_norm,
                    "Gold Answer": batch_gold_answers[i],
                    "Correct Answer": correct_answer,
                    "Gold Retrieved Docs": gold_retrieved_docs[i],
                    "Retrieved Titles So Far": unique_titles,
                    "Correct Retrieval": correct_retrieval,
                }
            )

    if write_csv and checkpoint_file:
        save_results_to_csv(pd.DataFrame(results, columns=CSV_COLUMNS), checkpoint_file)

    return results


def run_system(
    input_file,
    output_file,
    question_type,
    log_file,
    dataset_name,
    batch_size=8,
    max_turns=4,
    gpt=False,
    top_docs=6,
    remove_repeat_docs=False,
    start_index=0,
    end_index=5000,
):
    """Run the end-to-end evaluation loop."""
    data = pd.read_csv(input_file)
    total_rows = len(data)
    start = max(0, int(start_index or 0))
    end = total_rows if end_index is None else min(total_rows, int(end_index))

    if start >= end:
        print(f"[Subset] Empty slice: start={start}, end={end}, total={total_rows}")
        return

    data = data.iloc[start:end].reset_index(drop=True)
    print(f"[Subset] Using rows [{start}, {end}) -> {len(data)} rows")

    for offset in range(0, len(data), batch_size):
        batch = data.iloc[offset : offset + batch_size]

        batch_ids = [row["ID"] for _, row in batch.iterrows()]
        batch_questions = [row["Question"] for _, row in batch.iterrows()]
        batch_gold_answers = [_load_gold_answers(row) for _, row in batch.iterrows()]
        batch_gold_docs = [_load_gold_documents(row) for _, row in batch.iterrows()]

        task_contents = [
            f"Question: {question}\n" if gpt else f"Question: {question}\nContext:\n"
            for question in batch_questions
        ]

        batch_results = main_batch(
            task_contents=task_contents,
            question_type=question_type,
            ids=batch_ids,
            batch_gold_answers=batch_gold_answers,
            gold_retrieved_docs=batch_gold_docs,
            checkpoint_file=output_file,
            max_turns=max_turns,
            dataset_name=dataset_name,
            gpt=gpt,
            top_docs=top_docs,
            remove_repeat_docs=remove_repeat_docs,
            batch_questions=batch_questions,
            write_csv=False,
        )

        if batch_results:
            save_results_to_csv(pd.DataFrame(batch_results, columns=CSV_COLUMNS), output_file)

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    if len(df):
        df_sorted = df.sort_values(["ID", "Turn"])
        df_final = df_sorted.groupby("ID", as_index=False).tail(1)
    else:
        df_final = pd.DataFrame(columns=CSV_COLUMNS)

    acc = float(df_final["Correct Answer"].mean()) if len(df_final) else 0.0
    ret = float(df_final["Correct Retrieval"].mean()) if len(df_final) else 0.0
    avg_turn = float(df_final["Turn"].mean()) if len(df_final) else 0.0

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n===== Final Evaluation =====\n")
        f.write(f"Num rows (all turns): {len(df)}\n")
        f.write(f"Num questions (final rows): {len(df_final)}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Correct Answer (mean): {acc:.4f}\n")
        f.write(f"Correct Retrieval (mean): {ret:.4f}\n")
        f.write(f"Average Turn: {avg_turn:.2f}\n")
        f.write("============================================\n")

    print("Generation complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-turn RAG with a sufficiency gate and evidence selector."
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hotpotqa", "triviaqa", "2wikimultihopqa"],
        default=None,
    )
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--question_type", type=str, choices=["MCQ", "OEQ", "MATH"], default="OEQ")
    parser.add_argument("--gpt", action="store_true", default=False)
    parser.add_argument("--top_docs", type=int, default=6)
    parser.add_argument("--remove_repeat_docs", action="store_true", default=False)
    return parser.parse_args()


def load_corpus_and_searcher(dataset_name):
    """Load corpus, retriever settings, and initialize the searcher."""
    corpus_path, settings_path = get_bm25_resource_paths(dataset_name)

    with open(corpus_path, "rb") as f:
        loaded_corpus = pickle.load(f)
    with open(settings_path, "rb") as f:
        settings = pickle.load(f)

    searcher_obj = SimpleSearcher(settings["index_dir"])
    searcher_obj.set_bm25(settings.get("k1", 0.9), settings.get("b", 0.4))
    return loaded_corpus, searcher_obj


def load_reasoner_and_gate(args):
    """Load the main reasoner and the sufficiency model (base model only, no LoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global MODEL

    if args.gpt:
        if not os.getenv("CHATANYWHERE_API_KEY"):
            raise ValueError("CHATANYWHERE_API_KEY not set.")
        MODEL = "gpt-4o-mini"
        loaded_tokenizer = None
        loaded_model = None
    else:
        model_id = os.getenv("LLAMA_PATH")
        if not model_id:
            raise ValueError("LLAMA_PATH not set.")

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if loaded_tokenizer.pad_token is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        loaded_model.config.pad_token_id = loaded_tokenizer.pad_token_id

    suff_base_path = os.getenv("SUFF_BASE_PATH")
    if not suff_base_path:
        raise ValueError("SUFF_BASE_PATH must be set.")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    loaded_gate_tokenizer = AutoTokenizer.from_pretrained(
        suff_base_path,
        padding_side="left",
        use_fast=False,
    )
    if loaded_gate_tokenizer.pad_token is None:
        loaded_gate_tokenizer.pad_token = loaded_gate_tokenizer.eos_token

    gate_base_model = AutoModelForCausalLM.from_pretrained(
        suff_base_path,
        torch_dtype=compute_dtype,
    )

    loaded_gate_model = gate_base_model
    loaded_gate_model.config.pad_token_id = loaded_gate_tokenizer.pad_token_id
    print(f"[INFO] Using base sufficiency model from {suff_base_path} (no LoRA loaded)")

    if args.gpt:
        loaded_gate_model = accelerator.prepare(loaded_gate_model)
    else:
        loaded_model, loaded_gate_model = accelerator.prepare(loaded_model, loaded_gate_model)
        loaded_model.eval()

    loaded_gate_model.eval()
    return loaded_tokenizer, loaded_model, loaded_gate_tokenizer, loaded_gate_model


def main():
    global accelerator, tokenizer, model, gate_tokenizer, gate_model, searcher, corpus

    args = parse_args()

    if args.input_path is None:
        args.input_path = f"data/original/{args.experiment_name}_dev.csv"
    if args.output_path is None:
        args.output_path = f"predictions/{args.experiment_name}_predictions.csv"
    if args.log_path is None:
        args.log_path = f"logs/{args.experiment_name}_log.txt"

    args.dataset_name = infer_dataset_name(
        dataset_name=args.dataset_name,
        input_path=args.input_path,
        experiment_name=args.experiment_name,
    )

    accelerator = Accelerator()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    corpus, searcher = load_corpus_and_searcher(args.dataset_name)
    tokenizer, model, gate_tokenizer, gate_model = load_reasoner_and_gate(args)

    run_system(
        input_file=args.input_path,
        output_file=args.output_path,
        question_type=args.question_type,
        log_file=args.log_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_turns=args.max_turns,
        gpt=args.gpt,
        top_docs=args.top_docs,
        remove_repeat_docs=args.remove_repeat_docs,
        start_index=args.start_index,
        end_index=args.end_index,
    )


if __name__ == "__main__":
    main()

formatting2 = '''Respond only with the following format, nothing else:
Answer: [Provide the answer here]
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
'''

force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. "
    "Based on the context provided (if any), answer the following "
    "multihop question. Provide a brief multihop explanation. "
    "Prefer the shortest exact answer span that fully answers the question. "
) + formatting2

SUFF_SYSTEM_PROMPT = """You are a QA/RAG sufficiency judge.
Given a QUESTION and a CONTEXT (documents retrieved so far), 
decide whether the CONTEXT alone contains enough information to reliably answer the QUESTION. 
If not, list the gap items that describe what information is still missing.

You MUST respond with a single JSON object with the following shape:

{
  "sufficient": true/false,
  "gap_items": [
    {
      "category": "bridge_entity | attribute | relation | evidence_span | other",
      "target": "string",
      "slot": "string",
      "description": "string"
    },
    ...
  ]
}

If the information is sufficient, "gap_items" MUST be an empty list [].
"""


SELECTOR_SYSTEM_PROMPT = """
You are a sentence-level evidence selector for a multi-hop RAG system.

You will receive:
1. an ORIGINAL QUESTION,
2. MISSING FACTS that describe what information is still missing,
3. a numbered list of SENTENCES from retrieved documents.

Your task is to select the sentence ids that maximize answerability for the ORIGINAL QUESTION.

Selection policy:
1. First prioritize sentences that fill the MISSING FACTS, especially bridge entities, attributes, relations, and evidence spans needed for the next hop.
2. Then prioritize sentences that directly support the final answer to the ORIGINAL QUESTION.
3. Prefer sentences that are self-contained and explicit:
   - they mention the key entity, relation, attribute, date, number, or answer-bearing fact;
   - they remain understandable when extracted alone.
4. If a selected sentence depends on nearby context to be understandable or useful, include the minimal additional sentence(s) needed to preserve that context.
5. Do not infer, rewrite, paraphrase, or generate evidence text. Only return ids from the provided list.
6. If no sentence is useful, return an empty list.

Output format (strict):
Return exactly one JSON object and nothing else:
{"evidence_global_ids": [1, 5, 7]}

Constraints:
- "evidence_global_ids" must be a JSON array of integers.
- Select at most K sentences, where K is given in the user message.
- Only use ids that appear in the numbered sentence list.
- Do not repeat ids.
""".strip()

TRIVIAQA_SELECTOR_SYSTEM_PROMPT = """
You are a sentence-level evidence selector for a factoid QA system.

You will receive:
1. an ORIGINAL QUESTION,
2. MISSING FACTS that may help when direct answer evidence is unavailable,
3. a numbered list of SENTENCES from retrieved documents.

Your task is to select the sentence ids that give the most direct and minimal evidence for answering the ORIGINAL QUESTION.

Selection policy:
1. First prioritize sentences that directly answer the ORIGINAL QUESTION.
2. Prefer explicit answer-bearing sentences that state the target entity, date, location, number, role, or other fact asked in the question.
3. Prefer the smallest sufficient evidence set. If one sentence already answers the question clearly, select only that sentence.
4. Use MISSING FACTS only as backup guidance when no direct answer sentence is available.
5. Avoid generic background, broad biography, discography, list-page, or catalog-style sentences unless they directly answer the question.
6. If multiple sentences support the same answer, choose the most explicit and self-contained one.
7. If a selected sentence depends on nearby context, include only the minimal extra sentence(s) needed to preserve meaning.
8. Do not infer, paraphrase, or generate evidence text. Only return ids from the provided list.
9. If no sentence is useful, return an empty list.

Output format (strict):
Return exactly one JSON object and nothing else:
{"evidence_global_ids": [1, 5, 7]}

Constraints:
- "evidence_global_ids" must be a JSON array of integers.
- Select at most K sentences, where K is given in the user message.
- Only use ids that appear in the numbered sentence list.
- Do not repeat ids.
""".strip()

TRIVIAQA_FORCE_ANSWER_PROMPT = (
    "You are a factoid question-answering assistant. "
    "Use only the provided evidence. "
    "Answer with the shortest exact answer span that fully answers the question. "
    "Prefer the canonical alias written in the evidence, not the page title. "
    "Avoid extra descriptors, appositives, dates, locations, parenthetical disambiguation, or full sentences when a shorter exact answer is sufficient.\n"
    "Respond only with the following format, nothing else:\n"
    "Answer: [Provide the answer here]\n"
    "Rationale: [Provide the rationale here]\n\n"
    "Do not include any additional text, headers, or explanations outside this format.\n"
)
# S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA

[![ACL 2026](https://img.shields.io/badge/ACL-2026-8A2BE2.svg)](https://aclanthology.org/2026.acl-long.1185/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.23783-b31b1b.svg)](https://arxiv.org/abs/2604.23783)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This repository contains the official implementation of **S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA**, published as a long paper at **ACL 2026**.

[[ACL Anthology](https://aclanthology.org/2026.acl-long.1185/)]
[[arXiv](https://arxiv.org/abs/2604.23783)]
[[Code](https://github.com/nianaaa/S2G-RAG)]

## Overview

S2G-RAG is an iterative retrieval-augmented QA framework with an explicit and modular controller, **S2G-Judge**. At each retrieval turn, the judge reads only the question and the accumulated evidence context, then predicts:

1. whether the current evidence is sufficient to answer the question; and
2. structured **gap items** describing what information is still missing.

When the evidence is insufficient, the gap items are converted into the next retrieval query. A sentence-level Evidence Extractor then selects salient evidence from the retrieved documents, limiting context growth and reducing distractor interference. The answer reasoner is invoked only after the evidence is judged sufficient or the retrieval budget is exhausted.


## Prerequisites

- Linux is recommended.
- Python 3.10 or later.
- A CUDA-capable GPU is strongly recommended for local inference and LoRA training.
- A working Java runtime is required by Pyserini/Lucene when building and using BM25 indexes.
- Enough local storage for the QA datasets, retrieval corpora, Lucene indexes, and optional E5 embedding caches.

## Installation

1. Clone the repository.

```bash
git clone https://github.com/nianaaa/S2G-RAG.git
cd S2G-RAG
```

2. Create an environment.

```bash
conda create -n s2g-rag python=3.10 -y
conda activate s2g-rag
```

3. Install PyTorch for your CUDA environment, then install the remaining dependencies.

```bash
pip install \
  transformers accelerate peft trl datasets \
  pandas numpy scikit-learn openai \
  pyserini pysbd beir tqdm sentencepiece protobuf
```

For faster dense retrieval, install FAISS optionally:

```bash
pip install faiss-cpu
```

Ensure you have all the required dependencies installed (refer to requirements.txt or installation instructions in the repo).

4. Create output directories.

```bash
mkdir -p predictions logs supervision checkpoints
```

## Model and API Configuration

### Local models

The local inference scripts use environment variables for the answer reasoner, the base S2G-Judge model, and the trained LoRA adapter.

```bash
export LLAMA_PATH="/path/to/Meta-Llama-3-8B-Instruct"
export SUFF_BASE_PATH="/path/to/Llama-3.2-3B-Instruct"
export SUFF_LORA_PATH="/path/to/s2g-judge-lora"
```

### Teacher labeling API

`training/label_with_teacher.py` uses an OpenAI-compatible API. By default, it reads `OPENAI_API_KEY`; an alternative endpoint can be supplied with `OPENAI_BASE_URL` or `--base-url`.

```bash
export OPENAI_API_KEY="your-api-key"
# Optional:
# export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

### Optional GPT reasoner mode

Passing `--gpt` to the current generation or inference scripts switches the answer reasoner and Evidence Extractor to the code's GPT-compatible path. The checked-in implementation currently expects:

```bash
export OPENAI_API_KEY="your-api-key"
```

Edit the `base_url` in the generation and inference scripts when using a different OpenAI-compatible service. Without `--gpt`, the scripts use the local model specified by `LLAMA_PATH`.

## Prepare the QA Data

The preprocessing scripts create CSV files under `data/original/` with the fields needed by the generation and inference pipelines.

```bash
python data/prepare_hotpotqa.py
python data/prepare_triviaqa.py
python data/prepare_2wikimultihopqa.py
```

## Build Retrieval Resources

Build the dataset-specific BM25 corpora and Lucene indexes:

```bash
python bm25_search/bm25_init_hotpotqa.py
python bm25_search/bm25_init_triviaqa.py
python bm25_search/bm25_init_2wikiqa.py
```

## Quick Start: Inference with a Trained S2G-Judge

Set `SUFF_LORA_PATH` to a trained adapter before running inference.

### BM25 retrieval

```bash
export SUFF_LORA_PATH="/path/to/s2g-judge-lora"

python inference/inference_bm25.py \
  --experiment_name hotpotqa_s2g_bm25 \
  --dataset_name hotpotqa \
  --input_path data/original/hotpotqa_dev.csv \
  --output_path predictions/hotpotqa_s2g_bm25_predictions.csv \
  --log_path logs/hotpotqa_s2g_bm25_log.txt \
  --batch_size 8 \
  --max_turns 4 \
  --top_docs 6 \
  --remove_repeat_docs
```

### E5 dense retrieval

```bash
export SUFF_LORA_PATH="/path/to/s2g-judge-lora"
export E5_MODEL_NAME="intfloat/e5-base-v2"

python inference/inference_e5.py \
  --experiment_name hotpotqa_s2g_e5 \
  --dataset_name hotpotqa \
  --input_path data/original/hotpotqa_dev.csv \
  --output_path predictions/hotpotqa_s2g_e5_predictions.csv \
  --log_path logs/hotpotqa_s2g_e5_log.txt \
  --batch_size 8 \
  --max_turns 4 \
  --top_docs 6 \
  --remove_repeat_docs
```

The E5 implementation uses `intfloat/e5-base-v2` by default and caches corpus embeddings under `bm25_search/corpus/`. If FAISS is available, it is used for dense search; otherwise, the script falls back to NumPy similarity search.

### Important runtime arguments

| Argument | Description | Default |
|---|---|---:|
| `--dataset_name` | `hotpotqa`, `triviaqa`, or `2wikimultihopqa` | inferred when possible |
| `--batch_size` | Questions processed per batch | `8` |
| `--max_turns` | Maximum retrieval turns | `4` |
| `--top_docs` | Documents retrieved per turn | `6` |
| `--remove_repeat_docs` | Avoid retrieving the same document again across turns | off |
| `--start_index` | Inclusive start row | `0` |
| `--end_index` | Exclusive end row | `5000` |
| `--question_type` | `OEQ`, `MCQ`, or `MATH` | `OEQ` |
| `--gpt` | Use the GPT-compatible reasoner path | off |

For a smoke test, add `--end_index 32`. For a larger or full split, set `--end_index` to the desired exclusive upper bound.

## Train S2G-Judge from Execution Traces

The training pipeline follows six stages:

1. collect fixed-budget multi-turn execution traces;
2. convert trace rows into turn-level features;
3. label each snapshot with a stronger teacher;
4. remove conflicting or malformed supervision;
5. merge supervision from one or more datasets; and
6. LoRA-fine-tune the S2G-Judge backbone.

The commands below use HotpotQA as an example. Repeat stages 1–4 for TriviaQA and 2WikiMultiHopQA when building a multi-dataset judge.

### 1. Collect execution traces

`generation/generation.py` intentionally uses the base judge without loading a LoRA adapter and rolls every example to the fixed retrieval budget. This produces realistic intermediate evidence states for process supervision.

### 2. Build turn-level features

```bash
python training/build_features.py
```

Optional thresholds:

```text
--strong-pos-threshold 0.7
--strong-neg-threshold 0.1
```

### 3. Label snapshots with the teacher

```bash
python training/label_with_teacher.py
```

### 4. Filter and clean supervision

```bash
python training/filter_supervision.py
```

Optional balancing controls include:

```text
--balance-mode {none,downsample,upsample}
--balance-scope {all,hard_only}
--pos-neg-ratio 1.0
--balance-by-turn
```

### 5. Optional Merge supervision

```bash
python training/merge_supervision.py
```

### 6. LoRA fine-tuning

```bash
python training/train_s2g_judge_lora.py \
  --data supervision/s2g_train.jsonl \
  --base-model "$SUFF_BASE_PATH" \
  --output-dir checkpoints/s2g-judge \
  --bf16
```

Use `--fp16` instead of `--bf16` on hardware without BF16 support. If neither option is passed, the script defaults to FP16.

The final adapter is saved under:

```text
checkpoints/s2g-judge/lora/
```

Use it for inference with:

```bash
export SUFF_LORA_PATH="$PWD/checkpoints/s2g-judge/lora"
```

## Evaluation

Prediction files store one row per question and turn. The evaluator keeps the final row of each question trajectory and reports Exact Match and F1.

```bash
python evaluate_S2G-RAG.py \
  --experiment_name hotpotqa_s2g_bm25
```

The command above reads:

```text
predictions/hotpotqa_s2g_bm25_predictions.csv
```

The current evaluator selects the first **1,000 unique question IDs** before computing final-turn EM/F1. Modify the `keep_ids` slice in `evaluate_S2G-RAG.py` when a different evaluation size is required.

Fine-grained trajectory information is available in the prediction CSV, including:

- the accumulated Evidence Context;
- raw retrieved documents;
- S2G-Judge sufficiency outputs and structured gap items;
- generated answers and gold answers;
- retrieved titles and retrieval correctness; and
- the number of turns used by each trajectory.

Aggregate run statistics are also written to the selected log file.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li-etal-2026-s2g,
    title = "{S}2{G}-{RAG}: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented {QA}",
    author = "Li, Minghan and Zou, Junjie and Lv, Xinxuan and Zhang, Chao and Zhou, Guodong",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    year = "2026",
    doi = "10.18653/v1/2026.acl-long.1185",
    pages = "25846--25862"
}
```

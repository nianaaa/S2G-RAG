import os

DATASET_CONFIG = {
    "hotpotqa": {
        "wiki_corpus": False,
    },
    "triviaqa": {
        "wiki_corpus": False,
    },
    "2wikimultihopqa": {
        "wiki_corpus": True,
    },
}


def get_dataset_config(dataset_name):
    """Return retrieval settings for the selected dataset."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Expected one of: {', '.join(DATASET_CONFIG)}"
        )
    return DATASET_CONFIG[dataset_name]



def yes_no_to_bool(text, default=False):
    """Parse a yes/no input string with a default fallback."""
    text = (text or "").strip().lower()
    if not text:
        return default
    return text == "yes"



def build_generation_flags(gpt, remove_repeat_docs, use_wiki_corpus):
    """Build optional CLI flags for generation/inference."""
    flags = []

    if gpt:
        flags.append("--gpt")
    if remove_repeat_docs:
        flags.append("--remove_repeat_docs")
    if use_wiki_corpus:
        flags.append("--wiki_corpus")

    return " ".join(flags)



def generate_command_script(
    experiment_name,
    dataset_name,
    use_weighted_training=False,
    gpt=False,
    gpus="1,2,3",
    top_docs=2,
    remove_repeat_docs=True,
):
    """Generate a bash script for generation, training, and inference."""
    config = get_dataset_config(dataset_name)
    generation_flags = build_generation_flags(
        gpt=gpt,
        remove_repeat_docs=remove_repeat_docs,
        use_wiki_corpus=config["wiki_corpus"],
    )

    sh_content = f"""#!/bin/bash
set -e

mkdir -p nohup_logs

log_file=\"nohup_logs/{experiment_name}_command_log.txt\"

run_command() {{
    local cmd=\"$1\"
    echo \"Running: $cmd\" | tee -a \"$log_file\"
    if ! bash -c \"$cmd\"; then
        echo \"Error: Command failed - $cmd\" | tee -a \"$log_file\"
        exit 1
    fi
    echo \"Finished: $cmd\" | tee -a \"$log_file\"
}}

run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 generation/generation.py --experiment_name {experiment_name} --dataset_name {dataset_name} --input_path data/original/{dataset_name}_train.csv --top_docs {top_docs} {generation_flags}'

run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 training/prepare_training.py --experiment_name {experiment_name}'

run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 training/main.py --experiment_name {experiment_name}{" --weighted_training" if use_weighted_training else ""}'

run_command 'CUDA_VISIBLE_DEVICES={gpus} python3 inference/inference.py --experiment_name {experiment_name} --dataset_name {dataset_name} --input_path data/original/{dataset_name}_test.csv --top_docs {top_docs} {generation_flags}'
"""

    os.makedirs("bash_scripts", exist_ok=True)
    script_filename = f"run_{experiment_name}.sh"
    script_path = os.path.join("bash_scripts", script_filename)

    with open(script_path, "w", encoding="utf-8") as file:
        file.write(sh_content)

    print(f"{script_filename} has been created in bash_scripts.")
    print()
    print("Example usage:")
    print(f"chmod +x {script_path}")
    print(f"./{script_path}")
    print()
    print(f"Command logs: nohup_logs/{experiment_name}_command_log.txt")
    print(f"Detailed logs: logs/{experiment_name}_.txt")
    print(f"Predictions: predictions/{experiment_name}_predictions.csv")



def main():
    experiment_name = input(
        "Enter your custom S2G-RAG experiment name, for example 'S2G-RAG-test1': "
    ).strip()

    dataset_name = input(
        "Enter the dataset name (hotpotqa, triviaqa, 2wikimultihopqa): "
    ).strip()

    gpt = yes_no_to_bool(input("Use GPT? (yes/no): "), default=False)

    gpus = input("Enter CUDA visible devices (default: 1,2,3): ").strip()
    if not gpus:
        gpus = "1,2,3"

    use_weighted_training = yes_no_to_bool(
        input(
            "Optional settings. Press Enter to use defaults.\n"
            "Use weighted training? (yes/no, default: no): "
        ),
        default=False,
    )

    top_docs_input = input("Enter the number of top docs to retrieve (default: 2): ").strip()
    top_docs = int(top_docs_input) if top_docs_input else 2

    remove_repeat_docs = yes_no_to_bool(
        input("Remove repeated retrieved docs? (yes/no, default: yes): "),
        default=True,
    )

    generate_command_script(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        use_weighted_training=use_weighted_training,
        gpt=gpt,
        gpus=gpus,
        top_docs=top_docs,
        remove_repeat_docs=remove_repeat_docs,
    )


if __name__ == "__main__":
    main()

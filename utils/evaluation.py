import collections
import re
import string
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from utils.text_processing import safe_literal_eval


def pick_col(df, *candidates):
    """Pick the first matching column name with exact or relaxed matching."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    def normalize_name(name):
        return " ".join(str(name).strip().lower().replace("_", " ").split())

    normalized_map = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        normalized_candidate = normalize_name(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    return None


def ensure_list(value):
    """Convert a scalar or serialized list into a Python list."""
    parsed = safe_literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    if parsed is None:
        return []
    return [parsed]


def normalize_answer(text):
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punctuation(value):
        return "".join(ch for ch in value if ch not in string.punctuation)

    def lower(value):
        return value.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(str(text)))))


def exact_match_multiple(prediction, ground_truths):
    """Return 1 if the prediction exactly matches any ground truth answer."""
    normalized_prediction = normalize_answer(prediction)

    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    for truth in ground_truths:
        if normalized_prediction == normalize_answer(truth):
            return 1
    return 0


def extract_title_and_content(text):
    """Extract title and content from a formatted retrieval string."""
    title_match = re.search(r"Title:\s*(.*)\n", text or "", re.DOTALL)
    content_match = re.search(r"Content:\s*(.*)", text or "", re.DOTALL)

    title = title_match.group(1).strip() if title_match else None
    content = content_match.group(1).strip() if content_match else None
    return title, content


def exact_match_retrieval(prediction, ground_truths):
    """Return exact-match flags for retrieved title and content."""
    title, content = extract_title_and_content(prediction)
    normalized_pred_title = normalize_answer(title)
    normalized_pred_text = normalize_answer(content)

    result = [0, 0]
    for truth in ground_truths:
        normalized_title = normalize_answer(truth[0])
        if normalized_title == normalized_pred_title:
            result[0] = 1

        if len(truth) > 1:
            normalized_text = normalize_answer(truth[1])
            if normalized_text == normalized_pred_text:
                result[1] = 1

    return result


def get_tokens(text):
    """Tokenize normalized text for F1 computation."""
    if not text:
        return []
    return normalize_answer(text).split()


def compute_f1_multiple(prediction, gold_answers):
    """Compute the best token-level F1 against multiple gold answers."""
    pred_tokens = get_tokens(prediction)
    gold_token_lists = [get_tokens(answer) for answer in gold_answers]

    best_f1 = 0
    for gold_tokens in gold_token_lists:
        common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
        num_same = sum(common.values())

        if len(gold_tokens) == 0 or len(pred_tokens) == 0:
            best_f1 = max(best_f1, int(gold_tokens == pred_tokens))
        elif num_same != 0:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)

    return best_f1


def get_f1_score(df):
    """Compute average answer F1 over the provided rows."""
    total_f1 = 0
    total_predictions = len(df)

    for _, row in df.iterrows():
        prediction = row["Reasoner Answer"]
        gold_answers = ensure_list(row["Gold Answer"])
        total_f1 += compute_f1_multiple(prediction, gold_answers)

    return total_f1 / total_predictions if total_predictions > 0 else 0


def per_turn_generation(df):
    """Print verdict statistics by turn for generated data."""
    grouped = df.groupby("ID")
    group_sizes = grouped.size()

    turn_counts = df.groupby("Turn")["Verdict"].value_counts().unstack(fill_value=0)

    print("Verdict Distribution per Turn")
    print(turn_counts)
    print("\nGroup Size Distribution:")
    print(group_sizes.value_counts().sort_index())
    return turn_counts, None, None


def get_em_generation(df):
    """Compute final-turn EM and F1 for generated data."""
    last_entries = df.groupby("ID").tail(1)
    system_em = accuracy_score(last_entries["Verdict"], [1] * len(last_entries))

    print(f"EM: {system_em:.4f}")
    print(f"F1: {get_f1_score(last_entries):.4f}")
    return system_em


def get_em(df):
    """Compute final-turn EM and F1 for system predictions."""
    last_entries = df.groupby("ID").tail(1)
    system_em = accuracy_score(last_entries["Correct Answer"], [1] * len(last_entries))

    print(f"Total: {len(last_entries)}")
    print(f"EM: {system_em:.5f}")
    print(f"F1: {get_f1_score(last_entries):.5f}")
    return system_em


def custom_evaluation(df, name=""):
    """Compute answer metrics and optional critic metrics."""
    del name

    last_entries = df.groupby("ID").tail(1)
    system_em = accuracy_score(last_entries["Correct Answer"], [1] * len(last_entries))
    answer_f1 = get_f1_score(last_entries)

    critic_col = pick_col(df, "Critic Output", "Verdict")
    correct_col = pick_col(df, "Correct Answer")

    if correct_col is None:
        raise ValueError(
            f"Missing correctness label column (for example 'Correct Answer'). "
            f"Available columns: {list(df.columns)}"
        )

    if critic_col is None:
        results = {
            "Num Data Points": len(df),
            "EM": system_em,
            "F1": answer_f1,
            "Critic Metrics": "No critic column found (expected one of: Critic Output / Verdict)",
        }
        print("\n===== Evaluation Results =====")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        print("===================================================")
        return results

    critic_accuracy = accuracy_score(df[correct_col], df[critic_col])
    critic_precision = precision_score(df[correct_col], df[critic_col], zero_division=0)
    critic_recall = recall_score(df[correct_col], df[critic_col], zero_division=0)
    critic_f1 = f1_score(df[correct_col], df[critic_col], zero_division=0)

    tn, fp, fn, tp = confusion_matrix(df[correct_col], df[critic_col], labels=[0, 1]).ravel()
    critic_false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    critic_true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    results = {
        "Num Data Points": len(df),
        "EM": system_em,
        "F1": answer_f1,
        "Critic Accuracy": critic_accuracy,
        "Critic Precision": critic_precision,
        "Critic Recall": critic_recall,
        "Critic F1 Score": critic_f1,
        "Critic False Positive Rate": critic_false_positive_rate,
        "Critic True Negative Rate": critic_true_negative_rate,
    }

    print("\n===== Evaluation Results =====")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    print("===================================================")

    return results


def measure_redundancy(df, overlap_threshold=20):
    """Measure answer and rationale redundancy within each ID group."""
    total_answers = 0
    total_rationales = 0
    redundant_answers = 0
    redundant_rationales = 0

    answer_redundancy_counts = Counter()
    rationale_redundancy_counts = Counter()

    grouped = df.groupby("ID")

    for _, group in grouped:
        answer_counts = group["Reasoner Answer"].value_counts()
        group_size = len(group)
        total_answers += group_size

        group_redundant_answers = sum(count for count in answer_counts if count > 1)
        redundant_answers += group_redundant_answers
        answer_redundancy_counts[group_redundant_answers] += 1

        rationale_texts = group["Reasoner Rationale"].dropna().tolist()
        total_rationales += len(rationale_texts)

        group_redundant_rationales = 0
        if len(rationale_texts) > 1:
            vectorizer = CountVectorizer().fit_transform(rationale_texts)
            vectors = vectorizer.toarray()

            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    word_overlap = (vectors[i] & vectors[j]).sum()
                    if word_overlap >= overlap_threshold:
                        group_redundant_rationales += 1
                        break

            redundant_rationales += group_redundant_rationales
            rationale_redundancy_counts[group_redundant_rationales] += 1

        if group_redundant_rationales == 0:
            rationale_redundancy_counts[0] += 1

    answer_redundancy_percentage = (
        (redundant_answers / total_answers) * 100 if total_answers > 0 else 0
    )
    rationale_redundancy_percentage = (
        (redundant_rationales / total_rationales) * 100 if total_rationales > 0 else 0
    )

    print("\nOverall Redundancy:")
    print(f"Answer Redundancy: {answer_redundancy_percentage:.2f}%")
    print(
        f"Rationale Redundancy ({overlap_threshold} overlapping): "
        f"{rationale_redundancy_percentage:.2f}%"
    )

    def print_redundancy_counts(counts, label):
        print(f"\n{label}:")
        for key, value in sorted(counts.items(), reverse=True):
            print(f"  {value} group(s) with {key} repeated/overlapping items")

    print_redundancy_counts(answer_redundancy_counts, "Answer Redundancy Counts")
    print_redundancy_counts(
        rationale_redundancy_counts,
        f"Rationale Redundancy Counts ({overlap_threshold} overlapping)",
    )

    return {
        "Overall Answer Redundancy (%)": answer_redundancy_percentage,
        "Overall Rationale Redundancy (%)": rationale_redundancy_percentage,
        "Answer Redundancy Counts": answer_redundancy_counts,
        "Rationale Redundancy Counts": rationale_redundancy_counts,
    }


def evaluate_per_turn(df):
    """Print per-turn distributions for critic and answer correctness."""
    critic_col = pick_col(df, "Critic Output", "Verdict")
    correct_col = pick_col(df, "Correct Answer")

    print("Original Distribution:")
    print(len(df))

    if critic_col is not None:
        print(df[critic_col].value_counts())
    else:
        print("No critic column found (expected one of: Critic Output / Verdict)")

    if correct_col is not None:
        print(df[correct_col].value_counts())

    grouped = df.groupby("ID")
    group_sizes = grouped.size()

    print("\nGroup Size Distribution:")
    print(group_sizes.value_counts().sort_index())

    if critic_col is not None:
        critic_turn_counts = df.groupby("Turn")[critic_col].value_counts().unstack(fill_value=0)
        print("Critic counts per Turn:")
        print(critic_turn_counts)

    if correct_col is not None:
        correct_turn_counts = df.groupby("Turn")[correct_col].value_counts().unstack(fill_value=0)
        print("Correct Answer counts per Turn:")
        print(correct_turn_counts)


def compute_recall(df):
    """Compute average retrieval recall before the final turn."""
    recall_by_id = {}

    retrieved_column = "Retrieved Titles" if "Retrieved Titles" in df.columns else "Retrieved Title"
    gold_column = "Gold Retrieved Docs" if "Gold Retrieved Docs" in df.columns else "Gold Retrieved Doc"

    grouped = df.groupby("ID")
    empty_set_count = 0

    for id_value, group in grouped:
        group = group.iloc[:-1]

        all_retrieved_titles = set()
        for titles in group[retrieved_column]:
            all_retrieved_titles.update(ensure_list(titles))

        gold_documents = set()
        for titles in group[gold_column]:
            gold_documents.update(ensure_list(titles))

        if not all_retrieved_titles or not gold_documents:
            empty_set_count += 1
            continue

        common_docs = gold_documents.intersection(all_retrieved_titles)
        recall_by_id[id_value] = len(common_docs) / len(gold_documents) if gold_documents else 0.0

    average_recall = sum(recall_by_id.values()) / len(recall_by_id) if recall_by_id else 0.0
    no_retrieval = empty_set_count / len(grouped) if len(grouped) > 0 else 0.0
    num_single_row_groups = grouped.size()[grouped.size() == 1].count()

    print(
        f"Total data points ({len(grouped)}) = Total recall ({len(recall_by_id)}) + "
        f"(size of group size 1 ({num_single_row_groups}) = no retrieval attempt ({empty_set_count}))"
    )

    if len(grouped) != len(recall_by_id) + num_single_row_groups:
        print("Warning: Total data points do not equal total recall plus single-row groups.")

    if num_single_row_groups != empty_set_count:
        print("Warning: Single-row group count does not equal no-retrieval count.")

    print(f"Recall score: {average_recall:.4f}")
    print(f"Percentage retrieval is not attempted: {no_retrieval:.4f}")

    return average_recall, no_retrieval

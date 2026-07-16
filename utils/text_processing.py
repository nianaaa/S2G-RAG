import ast
import json
import re

ASSISTANT_BLOCK_PATTERNS = [
    re.compile(r"assistant\s*\n(.*)$", re.DOTALL | re.IGNORECASE),
]

NUMBER_AFTER_ANSWER_RE = re.compile(r"answer is [^\d]*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
ANY_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")

FINAL_ANSWER_RE = re.compile(r"Answer:\s*(.*?)\s*Rationale:", re.DOTALL)
FINAL_ANSWER_AND_RATIONALE_RE = re.compile(r"Answer:\s*(.*?)\s*Rationale:\s*(.*)", re.DOTALL)

SEARCH_QUERY_JSON_RE = re.compile(r'\{\s*"search_query":\s*".*?"\s*\}', re.DOTALL)
SEARCH_QUERY_SALVAGE_RE = re.compile(r"search_query.*?:(.*?)[,\n\r]+.*?reasoning", re.DOTALL)

ANSWER_LINE_RE = re.compile(r"(?i)^answer\s*:\s*(.+)$", re.M)
RATIONALE_LINE_RE = re.compile(r"(?i)^rationale\s*:\s*(.+)$", re.M)
MISSING_LINE_PATTERNS = [
    re.compile(r"(?i)^missing\s*information\s*:\s*(.+)$", re.M),
    re.compile(r"(?i)^missing\s*:\s*(.+)$", re.M),
]


def safe_literal_eval(value):
    """Safely parse a string representation of a Python literal."""
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip('"').strip("'")
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return value


def _normalize_json_key(key):
    """Normalize JSON keys so space/underscore variants are treated consistently."""
    return re.sub(r"[\s_]+", "", str(key).strip().lower())


def get_json_value(payload, *candidates, default=None):
    """Return the first matching JSON field, allowing space/underscore key variants."""
    if not isinstance(payload, dict):
        return default

    for candidate in candidates:
        if candidate in payload:
            return payload[candidate]

    normalized_map = {
        _normalize_json_key(key): value
        for key, value in payload.items()
    }
    for candidate in candidates:
        normalized_candidate = _normalize_json_key(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    return default


def extract_gap_items(payload):
    """Extract gap items from supported sufficiency-judge schemas."""
    return get_json_value(
        payload,
        "gap_items",
        "gap items",
        "missing_facts",
        default=[],
    )


def extract_evidence_global_ids(payload):
    """Extract evidence pointer ids from supported selector schemas."""
    return get_json_value(
        payload,
        "evidence_global_ids",
        "evidence global ids",
        default=[],
    )


def extract_assistant_output(text):
    """Extract the assistant portion from a chat-style output string."""
    if not text:
        return ""

    for pattern in ASSISTANT_BLOCK_PATTERNS:
        match = pattern.search(text)
        if match:
            output = match.group(1).strip()
            return re.sub(r"\n\s*\n", "\n", output)

    return text.strip()


def extract_number(text):
    """Extract the answer number from a math-style response."""
    if not text:
        return "Error: No number found"

    match = NUMBER_AFTER_ANSWER_RE.search(text)
    if match:
        return match.group(1).replace(",", "")

    matches = ANY_NUMBER_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")

    return "Error: No number found"


def extract_final_answer(text, question_type="OEQ"):
    """Extract the final answer only."""
    if question_type == "MATH":
        return extract_number(text)

    match = FINAL_ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip()

    return "Answer not found"


def extract_final_answer_and_rationale(text, question_type="OEQ"):
    """Extract the final answer and rationale."""
    if question_type == "MATH":
        return extract_number(text), text

    match = FINAL_ANSWER_AND_RATIONALE_RE.search(text or "")
    if match:
        answer = match.group(1).strip()
        rationale = match.group(2).strip()
        return answer, rationale

    return "Answer not found", "Rationale not found"


def parse_query(response):
    """Extract the search_query field from a JSON-like LLM response."""
    try:
        match = SEARCH_QUERY_JSON_RE.search(response)
        if match:
            parsed = json.loads(match.group(0))
            return parsed.get("search_query")

        salvage_match = SEARCH_QUERY_SALVAGE_RE.search(response)
        if salvage_match:
            return salvage_match.group(1).strip().strip('"')

        print("Attempt failed: 'search_query' and 'reasoning' key not found in response.")
        return response
    except json.JSONDecodeError:
        print("Attempt failed: json decode error")
        return response
    except Exception as exc:
        print(f"Attempt failed: unknown exception - {exc}")
        return response


def _parse_missing_field(text):
    """Parse a missing-information field into a list."""
    if not text:
        return []

    text = text.strip()

    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            pass

    return [item.strip() for item in re.split(r"[;,]\s*", text) if item.strip()]


def extract_answer_rationale_missing(text, question_type="OEQ"):
    """Return a tuple of (answer, rationale, missing_list)."""
    if not text:
        return "", "", []

    if question_type == "MATH":
        return extract_number(text), text, []

    raw = text.strip()

    try:
        parsed = json.loads(raw)
        answer = (parsed.get("answer", "") or "").strip()
        rationale = (parsed.get("rationale", "") or "").strip()

        missing = parsed.get(
            "missing",
            parsed.get("Missing information", parsed.get("Missing", [])),
        )

        if isinstance(missing, str):
            missing = parse_missing_field(missing)
        elif not isinstance(missing, list):
            missing = [missing]

        return answer, rationale, missing
    except Exception:
        pass

    answer_match = ANSWER_LINE_RE.search(raw)
    rationale_match = RATIONALE_LINE_RE.search(raw)

    missing_match = None
    for pattern in MISSING_LINE_PATTERNS:
        missing_match = pattern.search(raw)
        if missing_match:
            break

    answer = answer_match.group(1).strip() if answer_match else ""
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    missing_list = parse_missing_field(missing_match.group(1)) if missing_match else []

    return answer, rationale, missing_list

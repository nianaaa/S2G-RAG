import re


DEFAULT_E5_TITLE_PREFIX = "title:"
DEFAULT_E5_PASSAGE_PREFIX = "context:"
DEFAULT_E5_QUERY_PREFIX = "query:"
DEFAULT_E5_DOC_PREFIX = "passage:"
DEFAULT_E5_FORMAT_TAG = "kirag_title_context_v1"


def safe_model_tag(name: str) -> str:
    """Convert a free-form string into a filesystem-safe tag."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(name or ""))


def build_kirag_style_document(
    title: str,
    text: str,
    title_prefix: str = DEFAULT_E5_TITLE_PREFIX,
    passage_prefix: str = DEFAULT_E5_PASSAGE_PREFIX,
) -> str:
    """Format one document using KiRAG-style structured title/context fields."""
    clean_title = str(title or "").strip()
    clean_text = str(text or "").strip()

    if clean_title and clean_text:
        return f"{title_prefix} {clean_title}, {passage_prefix} {clean_text}"
    if clean_title:
        return f"{title_prefix} {clean_title}"
    if clean_text:
        return f"{passage_prefix} {clean_text}"
    return ""


def build_e5_query_text(query: str, query_prefix: str = DEFAULT_E5_QUERY_PREFIX) -> str:
    """Prefix a query string for E5-style retrieval."""
    return f"{query_prefix} {str(query or '').strip()}".strip()


def build_e5_passage_text(document_text: str, doc_prefix: str = DEFAULT_E5_DOC_PREFIX) -> str:
    """Prefix a structured document string for E5-style retrieval."""
    return f"{doc_prefix} {str(document_text or '').strip()}".strip()

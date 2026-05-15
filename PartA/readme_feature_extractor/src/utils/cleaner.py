import re

def extract_code_blocks(md_text):
    """Extract and separate code blocks from regular text."""
    code_blocks = []
    pattern = r'```[\w]*\n?(.*?)```'
    matches = re.findall(pattern, md_text, re.DOTALL)
    code_blocks.extend(matches)
    return code_blocks

def extract_tables(md_text):
    """Extract markdown tables as structured text."""
    table_pattern = r'(\|.+\|\n(?:\|[-:]+\|\n)(?:\|.+\|\n?)*)'
    tables = re.findall(table_pattern, md_text)
    return tables

def clean_readme(md_text):
    """
    Advanced cleaner: separates code blocks & tables from prose,
    strips badges/images/decorative links, collapses whitespace.
    Returns a dict with 'prose', 'code_blocks', and 'tables' keys.
    """
    if not md_text:
        return {"prose": "", "code_blocks": [], "tables": []}

    # 1. Extract code blocks before removing them
    code_blocks = extract_code_blocks(md_text)

    # 2. Extract tables before removing them
    tables = extract_tables(md_text)

    cleaned = md_text

    # 3. Remove HTML image tags
    cleaned = re.sub(r'<img[^>]+>', '', cleaned)

    # 4. Remove badge images wrapped in links  [![...](...)](...)
    cleaned = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', cleaned)

    # 5. Remove plain markdown images  ![alt](url)
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned)

    # 6. Remove fenced code blocks (keep prose only)
    cleaned = re.sub(r'```[\w]*\n?.*?```', '', cleaned, flags=re.DOTALL)

    # 7. Remove inline code that is very long (likely noise)
    cleaned = re.sub(r'`[^`]{80,}`', '', cleaned)

    # 8. Remove markdown tables
    cleaned = re.sub(r'(\|.+\|\n(?:\|[-:]+\|\n)(?:\|.+\|\n?)*)', '', cleaned)

    # 9. Strip decorative HTML tags (keep text content)
    cleaned = re.sub(r'<p[^>]*>', '', cleaned)
    cleaned = re.sub(r'</p>', '\n', cleaned)
    cleaned = re.sub(r'<h[1-6][^>]*>', '\n## ', cleaned)
    cleaned = re.sub(r'</h[1-6]>', '\n', cleaned)
    cleaned = re.sub(r'<div[^>]*>|</div>', '', cleaned)
    cleaned = re.sub(r'<br\s*/?>', '\n', cleaned)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)   # catch-all remaining tags

    # 10. Remove raw URLs that are not part of markdown links
    cleaned = re.sub(r'(?<!\()(https?://\S+)(?!\))', '', cleaned)

    # 11. Collapse excessive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return {
        "prose": cleaned.strip(),
        "code_blocks": code_blocks,
        "tables": tables,
    }


def build_llm_input(cleaned: dict, requirements_text: str = "") -> str:
    """
    Build a compact, structured prompt string from cleaned README data
    and optional requirements.txt content.
    """
    parts = []

    if cleaned["prose"]:
        parts.append("=== README (PROSE) ===\n" + cleaned["prose"][:6000])

    if cleaned["code_blocks"]:
        snippet = "\n---\n".join(cleaned["code_blocks"][:3])  # top 3 blocks
        parts.append("=== CODE SNIPPETS ===\n" + snippet[:1500])

    if cleaned["tables"]:
        parts.append("=== TABLES ===\n" + "\n\n".join(cleaned["tables"][:2]))

    if requirements_text and requirements_text.strip():
        parts.append("=== REQUIREMENTS.TXT ===\n" + requirements_text[:800])

    return "\n\n".join(parts)
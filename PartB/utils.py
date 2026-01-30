import fitz

def extract_text_from_pdf(file_or_path):
    """
    Extract text from a PDF file.
    
    Args:
        file_or_path: Path to PDF or file object.
    
    Returns:
        Extracted text as string.
    """
    # BUG: Fails if file_or_path is None
    pdf = fitz.open(file_or_path)
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

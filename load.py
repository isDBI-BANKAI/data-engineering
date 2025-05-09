import fitz
from pprint import pprint

from schema import Document, Page

def clean_content(content: [str]) -> [str]:
    cleaned_content = []
    last_text = None
    for text in content:
        if last_text is not None and last_text == text:
            continue
        cleaned_content.append(text)
        last_text = text
    return cleaned_content

def load_pdf(path: str):
    """
    Load a PDF file and return its content and metadata.

    Args:
        path (str): Path to the PDF file.

    Returns:
        str: the text content of the PDF.
    """
    doc = fitz.open(path)
    document = Document(pages=[])
    for page in doc:
        content = []
        text = page.get_text("text")
        content = [text.strip() for text in text.splitlines() if text.strip()]
        document.pages.append(Page(content=clean_content(content)))
    doc.close()
    return document

if __name__ == "__main__":
    path = "data/fas/4_Musharaka.PDF"
    doc = load_pdf(path)
    pprint(doc.model_dump())

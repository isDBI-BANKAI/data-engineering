import fitz  # PyMuPDF
import json
import os
import google.generativeai as genai
from typing import List

from keys import GEMINI_API_KEY
from config import GEMINI_MODEL_NAME

from prompt import get_prompt

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

PDF_PATH = "data/fas/4_Musharaka.PDF"
OUTPUT_PATH = "chunks_musharaka.json"
N_PAGES_PER_BATCH = 3

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

def clean_content(content: list[str]) -> str:
    cleaned_content = []
    last_text = None
    for text in content:
        if last_text is not None and last_text == text:
            continue
        cleaned_content.append(text)
        last_text = text
    return '\n'.join(cleaned_content)

def load_pdf(path: str):
    doc = fitz.open(path)
    pages = []
    for page in doc:
        content = []
        text = page.get_text("text")
        content = [text.strip() for text in text.splitlines() if text.strip()]
        pages.append(clean_content(content))
    doc.close()
    return pages

def extract_chunks_from_batch(text: str, context_to_keep: str) -> List[dict]:
    prompt = get_prompt(text, context_to_keep)
    response = model.generate_content(prompt).text
    try:
        return json.loads(response.strip("```json").strip("```"))
    except json.JSONDecodeError:
        print("JSON parsing failed. Skipping batch.")
        return []

def process_document_in_batches(pages: List[str], batch_size: int = N_PAGES_PER_BATCH):
    results = []
    context = ""
    total_pages = len(pages)

    for i in range(0, total_pages, batch_size):
        batch_pages = pages[i:i+batch_size]
        page_range_str = f"{i+1}-{i+len(batch_pages)}"
        text = "\n\n".join(batch_pages)

        print(f"Processing pages {page_range_str}...")
        output = extract_chunks_from_batch(text, context_to_keep=context)
        try:
            chunks = output.get("chunks", [])
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
        context = output.get("context_to_keep", "")
        if not chunks:
            print("No chunks found. Skipping batch.")
            continue

        results.extend(chunks)

    return results

def main():
    base_dir = "data/raw/ss"
    out_dir = "data/rag/ss"
    for file in os.listdir(base_dir):
        print("Loading PDF...", file)
        pages = load_pdf(os.path.join(base_dir, file))

        print("Processing in batches...")
        chunks = process_document_in_batches(pages, N_PAGES_PER_BATCH)

        output_path = os.path.join(out_dir, file.replace(".PDF", ".json"))
        print(f"Saving {len(chunks)} chunks to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

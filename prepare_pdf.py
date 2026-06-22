import os
from pathlib import Path
import pandas as pd
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "documents" / "Second-world-war.pdf"
OUTPUT_PATH = BASE_DIR / "data" / "embedded_ww2_chunks.parquet"

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def main():
    if not PDF_PATH.exists():
        print(f"❌ Could not find PDF at: {PDF_PATH}")
        return

    print("📄 Reading and extracting text from PDF...")
    reader = PdfReader(PDF_PATH)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "

    # 1. Chunking: Split text into ~500 character blocks with a small overlap
    print("✂️ Chunking text...")
    chunk_size = 500
    overlap = 100
    chunks = []
    
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)

    # 2. Build a Dataframe
    df = pd.DataFrame({"text_chunk": chunks})
    print(f"🧩 Created {len(df)} distinct text chunks.")

    # 3. Generate Embeddings
    print("🤖 Generating embeddings with OpenAI...")
    df['embedding'] = df['text_chunk'].apply(get_embedding)

    # 4. Save to Parquet
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"🎉 Success! Vector database created at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

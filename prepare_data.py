import os
import pandas as pd
import numpy as np
import kagglehub
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    print("🚀 Starting data preparation script...")

    # 1. Initialize OpenAI Client (reads OPENAI_API_KEY from environment)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("❌ Error: OPENAI_API_KEY not found in environment variables or .env file.")
    client = OpenAI()

    # 2. Download dataset using kagglehub
    print("📥 Downloading Amazon Fine Food Reviews dataset from Kaggle...")
    download_path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    csv_path = os.path.join(download_path, "Reviews.csv")

    # 3. Load and sample the data
    print("📄 Reading CSV and sampling 1,000 reviews...")
    df = pd.read_csv(csv_path)
    
    # Take a deterministic 1,000 review sample to keep it fast and budget-friendly
    df = df.head(1000).copy()
    
    # Create the combined text string for embedding context
    df['combined'] = "Summary: " + df.Summary.str.strip() + "; Review: " + df.Text.str.strip()

    # 4. Define embedding function
    def get_embedding(text, model="text-embedding-3-small"):
        if not isinstance(text, str):
            text = ""
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    # 5. Generate embeddings
    print("🤖 Requesting embeddings from OpenAI... (This might take a minute)")
    try:
        df['ada_embedding'] = df.combined.apply(get_embedding)
        print("✅ Embeddings successfully generated!")
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return

    # 6. Ensure data directory exists and save as Parquet
    os.makedirs('data', exist_ok=True)
    output_path = 'data/embedded_1k_reviews.parquet'
    
    print(f"💾 Saving processed data to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("🎉 All done! You can now start your Flask API using 'python app.py'.")

if __name__ == "__main__":
    main()

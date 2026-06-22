import numpy as np

from services.data_store import ReviewsStore, WW2Store
from services.openai_client import get_openai_client


def embed_text(text: str, model: str) -> np.ndarray:
    cleaned = text.replace("\n", " ")
    client = get_openai_client()
    embedding = client.embeddings.create(input=[cleaned], model=model).data[0].embedding
    return np.array(embedding).astype("float32")


def search_top_k(
    query: str,
    store: ReviewsStore | WW2Store,
    *,
    embedding_model: str,
    top_k: int,
    text_column: str,
) -> list[dict]:
    query_vector = embed_text(query, embedding_model)
    similarities = np.dot(store.embeddings_matrix, query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = store.dataframe.iloc[idx]
        results.append(
            {
                "index": int(idx),
                "score": float(similarities[idx]),
                "text": str(row[text_column]),
            }
        )
    return results


def search_reviews(query: str, store: ReviewsStore, *, embedding_model: str, top_k: int) -> list[dict]:
    query_vector = embed_text(query, embedding_model)
    similarities = np.dot(store.embeddings_matrix, query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = store.dataframe.iloc[idx]
        results.append(
            {
                "score": float(similarities[idx]),
                "summary": str(row["Summary"]),
                "review_text": str(row["Text"]),
            }
        )
    return results

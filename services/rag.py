from services.data_store import WW2Store
from services.embedding_search import search_top_k
from services.openai_client import get_openai_client

SYSTEM_PROMPT = (
    "You are an AI history assistant. Answer the user's question using ONLY "
    "the extracted document context provided below. If the answer cannot be found "
    "in the context, politely state that you don't know based on the provided material."
)


def ask_ww2_history(question: str, store: WW2Store, config) -> dict:
    matches = search_top_k(
        question,
        store,
        embedding_model=config["EMBEDDING_MODEL"],
        top_k=config["TOP_K"],
        text_column="text_chunk",
    )

    retrieved_context = "".join(f"- {match['text']}\n\n" for match in matches)
    user_prompt = f"Context from History PDF:\n{retrieved_context}\nQuestion: {question}"

    client = get_openai_client()
    response = client.responses.create(
        model=config["RAG_MODEL"],
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    return {
        "question": question,
        "answer": response.output_text,
        "sources_used": [match["text"] for match in matches],
    }

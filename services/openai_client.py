from openai import OpenAI

_client: OpenAI | None = None


def init_openai_client(api_key: str) -> OpenAI:
    global _client
    _client = OpenAI(api_key=api_key)
    return _client


def get_openai_client() -> OpenAI:
    if _client is None:
        raise RuntimeError("OpenAI client is not initialized")
    return _client

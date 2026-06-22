import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReviewsStore:
    dataframe: pd.DataFrame
    embeddings_matrix: np.ndarray


@dataclass
class WW2Store:
    dataframe: pd.DataFrame
    embeddings_matrix: np.ndarray


@dataclass
class DataStore:
    reviews: ReviewsStore | None
    ww2: WW2Store | None
    instructions_prompt: str


_store: DataStore | None = None


def init_data_store(
    reviews_path,
    ww2_path,
    instructions_path,
    *,
    require_reviews: bool = True,
) -> DataStore:
    global _store

    instructions_prompt = instructions_path.read_text(encoding="utf-8")

    reviews: ReviewsStore | None = None
    if reviews_path.exists():
        reviews_df = pd.read_parquet(reviews_path)
        reviews_matrix = np.array(reviews_df["ada_embedding"].tolist()).astype("float32")
        reviews = ReviewsStore(dataframe=reviews_df, embeddings_matrix=reviews_matrix)
    elif require_reviews:
        raise FileNotFoundError(f"Reviews data not found: {reviews_path}")
    else:
        logger.warning("Reviews data not found at %s", reviews_path)

    ww2: WW2Store | None = None
    try:
        if ww2_path.exists():
            ww2_df = pd.read_parquet(ww2_path)
            ww2_matrix = np.array(ww2_df["embedding"].tolist()).astype("float32")
            ww2 = WW2Store(dataframe=ww2_df, embeddings_matrix=ww2_matrix)
        else:
            logger.warning("WW2 data not found at %s", ww2_path)
    except Exception:
        logger.exception("Could not load WW2 data from %s", ww2_path)

    _store = DataStore(
        reviews=reviews,
        ww2=ww2,
        instructions_prompt=instructions_prompt,
    )
    return _store


def get_data_store() -> DataStore:
    if _store is None:
        raise RuntimeError("Data store is not initialized")
    return _store

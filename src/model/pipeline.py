from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from src.model.features import PairRecord, build_pair_text, build_structured_features


@dataclass
class FeatureArtifacts:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer


def build_feature_matrix_fit(records: list[PairRecord]) -> tuple[FeatureArtifacts, csr_matrix]:
    pair_texts = [build_pair_text(r.current_description, r.prior_description) for r in records]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        lowercase=False,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        lowercase=False,
    )

    word_x = word_vectorizer.fit_transform(pair_texts)
    char_x = char_vectorizer.fit_transform(pair_texts)
    structured_x = csr_matrix(build_structured_features(records), dtype=np.float32)

    matrix = hstack([word_x, char_x, structured_x], format="csr")
    return FeatureArtifacts(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer), matrix


def build_feature_matrix_transform(records: list[PairRecord], artifacts: FeatureArtifacts) -> csr_matrix:
    pair_texts = [build_pair_text(r.current_description, r.prior_description) for r in records]

    word_x = artifacts.word_vectorizer.transform(pair_texts)
    char_x = artifacts.char_vectorizer.transform(pair_texts)
    structured_x = csr_matrix(build_structured_features(records), dtype=np.float32)

    return hstack([word_x, char_x, structured_x], format="csr")

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .retrieval_demo import normalize_text


SAMPLE_QA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_qa.json"

STOPWORDS = {
    "anh",
    "bi",
    "cac",
    "cho",
    "co",
    "cua",
    "da",
    "dang",
    "de",
    "duoc",
    "gi",
    "hoi",
    "khi",
    "khong",
    "la",
    "mot",
    "nguoi",
    "phai",
    "sao",
    "thi",
    "toi",
    "trong",
    "va",
    "ve",
    "voi",
}


def load_sample_qa(path: Path = SAMPLE_QA_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1 and token not in STOPWORDS
    }


def _score(query: str, candidate: str) -> float:
    query_norm = normalize_text(query)
    candidate_norm = normalize_text(candidate)
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)

    if not query_tokens or not candidate_tokens:
        return SequenceMatcher(None, query_norm, candidate_norm).ratio()

    shared = query_tokens & candidate_tokens
    overlap = len(shared) / max(len(query_tokens), 1)
    jaccard = len(shared) / max(len(query_tokens | candidate_tokens), 1)
    char_score = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    return 0.50 * overlap + 0.30 * jaccard + 0.20 * char_score


def find_best_sample_qa(
    query: str,
    samples: list[dict[str, Any]],
    threshold: float = 0.50,
) -> dict[str, Any] | None:
    if not samples:
        return None

    best: dict[str, Any] | None = None
    best_score = -1.0
    for item in samples:
        score = _score(query, str(item.get("question", "")))
        if score > best_score:
            best = item
            best_score = score

    if best is None or best_score < threshold:
        return None
    matched = dict(best)
    matched["match_score"] = round(float(min(best_score, 1.0)), 4)
    return matched


def format_sample_answer(sample: dict[str, Any]) -> str:
    answer = " ".join(str(sample.get("answer", "")).split())
    legal_basis = " ".join(str(sample.get("legal_basis", "")).split())

    conclusion, explanation = answer, ""
    for delimiter in [". ", "; "]:
        if delimiter in answer:
            conclusion, explanation = answer.split(delimiter, 1)
            conclusion = conclusion.strip()
            explanation = explanation.strip()
            if delimiter == ". " and conclusion and not conclusion.endswith("."):
                conclusion += "."
            break

    lines = [
        f"* Kết luận {conclusion}",
    ]
    if explanation:
        lines.append(f"* Giải thích {explanation}")
    if legal_basis:
        lines.append(f"* Căn cứ {legal_basis}")
    return "\n".join(lines)

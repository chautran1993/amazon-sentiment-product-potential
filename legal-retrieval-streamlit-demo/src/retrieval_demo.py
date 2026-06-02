from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from math import log
from typing import Any

import pandas as pd

from .data_loader import article_display_name, article_text


STOPWORDS = {
    "a",
    "anh",
    "bi",
    "cac",
    "cach",
    "can",
    "cho",
    "co",
    "con",
    "cua",
    "da",
    "dang",
    "de",
    "den",
    "di",
    "duoc",
    "gi",
    "han",
    "hay",
    "hoi",
    "khi",
    "khong",
    "la",
    "lam",
    "mot",
    "nay",
    "neu",
    "nguoi",
    "nhu",
    "o",
    "phai",
    "ra",
    "sao",
    "thi",
    "toi",
    "trong",
    "tu",
    "ve",
    "vi",
    "voi",
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = text.replace("đ", "d")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def expand_query(text: str) -> str:
    normalized = normalize_text(text)
    expansions = {
        "co bau": "mang thai thai san lao dong nu",
        "bau": "mang thai thai san lao dong nu",
        "duoi viec": "cham dut hop dong lao dong sa thai don phuong",
        "bi duoi": "cham dut hop dong lao dong sa thai don phuong",
        "nghi viec": "cham dut hop dong lao dong tro cap thoi viec tro cap that nghiep",
        "mat viec": "tro cap mat viec tro cap that nghiep cham dut hop dong",
        "that nghiep": "bao hiem that nghiep tro cap that nghiep viec lam",
        "sep": "nguoi su dung lao dong cong ty",
        "cong ty": "nguoi su dung lao dong doanh nghiep",
        "lam them": "lam them gio tien luong lam them gio",
        "tang ca": "lam them gio tien luong lam them gio",
        "thu viec": "thu viec hop dong thu viec thoi gian thu viec",
        "luong": "tien luong tra luong",
        "bao hiem": "bao hiem xa hoi bao hiem that nghiep",
    }
    extra = []
    for phrase, words in expansions.items():
        if phrase in normalized:
            extra.append(words)
    return f"{text} {' '.join(extra)}".strip()


def _token_counter(text: str) -> Counter[str]:
    return Counter(
        token
        for token in normalize_text(text).split()
        if len(token) > 1 and token not in STOPWORDS
    )


def _jaccard(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 0.0


def _overlap_ratio(query_tokens: Counter[str], candidate_tokens: Counter[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    shared = set(query_tokens) & set(candidate_tokens)
    return len(shared) / max(len(query_tokens), 1)


def _bm25_question_scores(query_tokens: Counter[str], questions: list[str]) -> dict[str, float]:
    question_docs = [(question, _token_counter(question)) for question in questions]
    doc_freq = Counter()
    for _, tokens in question_docs:
        for token in tokens:
            doc_freq[token] += 1

    n_docs = max(len(question_docs), 1)
    avg_len = sum(sum(tokens.values()) for _, tokens in question_docs) / n_docs
    avg_len = max(avg_len, 1.0)
    k1 = 1.4
    b = 0.65
    scores = {}

    for question, tokens in question_docs:
        doc_len = max(sum(tokens.values()), 1)
        score = 0.0
        for term, query_tf in query_tokens.items():
            tf = tokens.get(term, 0)
            if not tf:
                continue
            idf = log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += query_tf * idf * (tf * (k1 + 1)) / denom
        scores[question] = score
    return scores


def find_closest_question(query: str, details: pd.DataFrame) -> tuple[str, float]:
    questions = details["question"].drop_duplicates().tolist()
    expanded_query = expand_query(query)
    query_tokens = _token_counter(expanded_query)
    best_question = questions[0]
    best_score = -1.0
    query_norm = normalize_text(expanded_query)
    bm25_scores = _bm25_question_scores(query_tokens, questions)
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0

    for question in questions:
        question_norm = normalize_text(question)
        question_tokens = _token_counter(question)
        jaccard_score = _jaccard(query_tokens, question_tokens)
        overlap_score = _overlap_ratio(query_tokens, question_tokens)
        char_score = SequenceMatcher(None, query_norm, question_norm).ratio()
        bm25_score = bm25_scores.get(question, 0.0) / max_bm25 if max_bm25 else 0.0
        phrase_boost = 0.08 if question_norm in query_norm or normalize_text(query) in question_norm else 0.0
        score = (
            0.45 * bm25_score
            + 0.25 * overlap_score
            + 0.20 * jaccard_score
            + 0.10 * char_score
            + phrase_boost
        )
        if score > best_score:
            best_question = question
            best_score = score

    return best_question, round(float(min(best_score, 1.0)), 4)


def build_results_table(
    question: str,
    method: str,
    top_k: int,
    details: pd.DataFrame,
    corpus: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    row = details[(details["question"] == question) & (details["method"] == method)]
    if row.empty:
        return pd.DataFrame()

    record = row.iloc[0]
    gold_set = set(record["gold_list"])
    rows = []
    for rank, article_id in enumerate(record["predicted_list"][:top_k], start=1):
        rows.append(
            {
                "rank": rank,
                "article_id": article_id,
                "score": round(1 / rank, 4),
                "method": method,
                "matched_gold": "Yes" if article_id in gold_set else "No",
                "article": article_display_name(article_id, corpus),
                "content": article_text(article_id, corpus),
            }
        )
    return pd.DataFrame(rows)


def live_bm25_search(
    query: str,
    top_k: int,
    corpus: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Small dependency-free BM25 search for arbitrary demo questions."""
    expanded_query = expand_query(query)
    query_terms = list(_token_counter(expanded_query).keys())
    if not query_terms:
        return pd.DataFrame()

    documents = []
    doc_freq = Counter()
    for article_id, item in corpus.items():
        text = " ".join(
            [
                str(item.get("tieu_de", "")),
                article_text(article_id, corpus),
            ]
        )
        term_counts = _token_counter(text)
        if not term_counts:
            continue
        documents.append((article_id, term_counts, sum(term_counts.values())))
        for term in term_counts:
            doc_freq[term] += 1

    n_docs = len(documents)
    avg_len = sum(doc_len for _, _, doc_len in documents) / max(n_docs, 1)
    k1 = 1.5
    b = 0.75

    scored = []
    for article_id, term_counts, doc_len in documents:
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if not tf:
                continue
            idf = log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scored.append((article_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    if not scored:
        scored = _fallback_article_search(expanded_query, corpus)

    rows = []
    for rank, (article_id, score) in enumerate(scored[:top_k], start=1):
        rows.append(
            {
                "rank": rank,
                "article_id": article_id,
                "score": round(float(score), 4),
                "method": "BM25_lite_live",
                "article": article_display_name(article_id, corpus),
                "content": article_text(article_id, corpus),
            }
        )
    return pd.DataFrame(rows)


def _fallback_article_search(
    query: str,
    corpus: dict[str, dict[str, Any]],
) -> list[tuple[str, float]]:
    query_norm = normalize_text(query)
    query_terms = set(query_norm.split())
    scored = []

    for article_id, item in corpus.items():
        title = str(item.get("tieu_de", ""))
        text = f"{title} {article_text(article_id, corpus)}"
        text_norm = normalize_text(text)
        if not text_norm:
            continue

        text_terms = set(text_norm.split())
        overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
        char_score = SequenceMatcher(None, query_norm[:500], text_norm[:1000]).ratio()
        score = 4 * overlap + char_score
        if score > 0:
            scored.append((article_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _shorten_text(text: str, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."


def generate_reference_answer(query: str, results: pd.DataFrame) -> str:
    if results.empty:
        return (
            "Chưa đủ căn cứ để tạo câu trả lời tham khảo. Hãy nhập câu hỏi cụ thể hơn "
            "hoặc tăng Top-K để hệ thống tìm thêm điều luật liên quan."
        )

    top_rows = results.head(3).to_dict("records")
    citations = [f"{row['article_id']} ({row['article']})" for row in top_rows]
    evidence = "\n\n".join(
        f"- {row['article_id']}: {_shorten_text(row['content'])}" for row in top_rows
    )

    return (
        "Câu trả lời tham khảo:\n\n"
        f"Dựa trên các điều luật được truy xuất, câu hỏi có khả năng liên quan nhiều nhất đến: "
        f"{'; '.join(citations)}.\n\n"
        "Tóm tắt căn cứ chính:\n\n"
        f"{evidence}\n\n"
        "Lưu ý: đây là câu trả lời demo sinh tự động từ Top-K điều luật, không thay thế tư vấn pháp lý."
    )


def build_comparison_table(
    question: str,
    top_k: int,
    details: pd.DataFrame,
    corpus: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for _, record in details[details["question"] == question].iterrows():
        method = record["method"]
        gold_set = set(record["gold_list"])
        predicted = record["predicted_list"][:top_k]
        hits = [article_id for article_id in predicted if article_id in gold_set]
        rows.append(
            {
                "method": method,
                "top_k": top_k,
                "hits_in_top_k": len(hits),
                "hit_labels": ", ".join(hits) if hits else "-",
                "top_1": article_display_name(predicted[0], corpus) if predicted else "-",
                "mrr": round(float(record.get("mrr", 0)), 4),
                "processing_time_sec": round(float(record.get("processing_time", 0)), 4),
                "recall_at_k": round(float(record.get(f"recall@{top_k}", 0)), 4)
                if f"recall@{top_k}" in record
                else None,
                "precision_at_k": round(float(record.get(f"precision@{top_k}", 0)), 4)
                if f"precision@{top_k}" in record
                else None,
            }
        )
    return pd.DataFrame(rows)


def gold_basis_for_question(question: str, eval_dataset: pd.DataFrame) -> dict[str, str]:
    row = eval_dataset[eval_dataset["question"] == question]
    if row.empty:
        return {"legal_basis": "", "gold_answer": "", "gold_labels": ""}
    record = row.iloc[0]
    return {
        "legal_basis": str(record.get("legal_basis", "")),
        "gold_answer": str(record.get("gold_answer", "")),
        "gold_labels": str(record.get("gold_labels", "")),
    }

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .data_loader import article_display_name
from .retrieval_demo import normalize_text


def _article_ids(results: pd.DataFrame, limit: int = 5) -> list[str]:
    if results.empty or "article_id" not in results:
        return []
    return results["article_id"].head(limit).dropna().astype(str).tolist()


def extract_query_keyphrases(
    query: str,
    keyphrases: list[dict[str, Any]],
    limit: int = 12,
) -> pd.DataFrame:
    query_norm = normalize_text(query)
    rows = []
    for item in keyphrases:
        if isinstance(item, dict):
            phrase = str(item.get("phrase") or item.get("keyphrase") or item.get("name") or "")
            article_id = item.get("article_id", "")
            source = item.get("source", "KB.Keyphrases")
            score = item.get("weight", item.get("confidence", 1.0))
        else:
            phrase = str(item)
            article_id = ""
            source = "KB.Keyphrases"
            score = 1.0
        if not phrase:
            continue
        phrase_norm = normalize_text(phrase)
        if phrase_norm and phrase_norm in query_norm:
            rows.append(
                {
                    "keyphrase": phrase,
                    "article_id": article_id,
                    "source": source,
                    "score": score,
                }
            )
    return pd.DataFrame(rows[:limit])


def concepts_for_articles(
    article_ids: list[str],
    article_nodes: dict[str, list[dict[str, Any]]],
    concepts: list[dict[str, Any]],
    limit: int = 15,
) -> pd.DataFrame:
    concept_by_name = {normalize_text(c.get("name", "")): c for c in concepts}
    rows = []
    seen = set()

    for article_id in article_ids:
        for node in article_nodes.get(article_id, []):
            if node.get("type") != "concept":
                continue
            name = str(node.get("name", ""))
            key = (article_id, normalize_text(name))
            if key in seen:
                continue
            seen.add(key)
            concept = concept_by_name.get(normalize_text(name), {})
            rows.append(
                {
                    "article_id": article_id,
                    "concept": name,
                    "concept_id": concept.get("concept_id", ""),
                    "source": node.get("source", "article_nodes"),
                    "confidence": node.get("confidence", ""),
                    "attributes": ", ".join(concept.get("attributes", [])[:6])
                    if isinstance(concept.get("attributes"), list)
                    else "",
                }
            )
            if len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def keyphrases_for_articles(
    article_ids: list[str],
    article_nodes: dict[str, list[dict[str, Any]]],
    limit: int = 18,
) -> pd.DataFrame:
    rows = []
    seen = set()
    for article_id in article_ids:
        for node in article_nodes.get(article_id, []):
            if node.get("type") != "keyphrase":
                continue
            name = str(node.get("name", ""))
            key = (article_id, normalize_text(name))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "article_id": article_id,
                    "keyphrase": name,
                    "source": node.get("source", "article_nodes"),
                    "confidence": node.get("confidence", ""),
                }
            )
            if len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def triples_for_articles(
    article_ids: list[str],
    triples: pd.DataFrame,
    limit: int = 15,
) -> pd.DataFrame:
    if triples.empty or "article_id" not in triples:
        return pd.DataFrame()
    df = triples[triples["article_id"].astype(str).isin(article_ids)].copy()
    if "confidence" in df:
        df = df.sort_values("confidence", ascending=False)
    columns = [
        col
        for col in [
            "article_id",
            "subject",
            "relation",
            "object",
            "relation_group",
            "confidence",
            "evidence_text",
        ]
        if col in df.columns
    ]
    return df[columns].head(limit)


def edges_for_articles(
    article_ids: list[str],
    edges: pd.DataFrame,
    limit: int = 15,
) -> pd.DataFrame:
    if edges.empty or "article_id" not in edges:
        return pd.DataFrame()
    df = edges[edges["article_id"].astype(str).isin(article_ids)].copy()
    if "weight" in df:
        df = df.sort_values("weight", ascending=False, na_position="last")
    columns = [
        col
        for col in [
            "article_id",
            "source",
            "relation",
            "target",
            "relation_group",
            "weight",
            "confidence",
            "edge_type",
        ]
        if col in df.columns
    ]
    return df[columns].head(limit)


def rules_for_query_and_articles(
    query: str,
    article_ids: list[str],
    rules: list[dict[str, Any]],
    triples: pd.DataFrame,
    limit: int = 8,
) -> pd.DataFrame:
    text = normalize_text(query + " " + " ".join(article_ids))
    relation_text = ""
    if not triples.empty and "article_id" in triples and "relation" in triples:
        relation_text = " ".join(
            triples[triples["article_id"].astype(str).isin(article_ids)]["relation"]
            .dropna()
            .astype(str)
            .head(200)
            .tolist()
        )
    text = f"{text} {normalize_text(relation_text)}"

    scored = []
    for rule in rules:
        haystack = normalize_text(
            " ".join(
                [
                    str(rule.get("name", "")),
                    str(rule.get("description", "")),
                    str(rule.get("formal", "")),
                ]
            )
        )
        score = sum(1 for token in set(text.split()) if token in haystack)
        if score > 0:
            scored.append((score, rule))

    if not scored:
        scored = [(1, rule) for rule in rules[:limit]]

    scored.sort(key=lambda item: item[0], reverse=True)
    return pd.DataFrame(
        [
            {
                "rule_id": rule.get("rule_id", ""),
                "name": rule.get("name", ""),
                "description": rule.get("description", ""),
                "formal": rule.get("formal", ""),
                "matched_score": score,
            }
            for score, rule in scored[:limit]
        ]
    )


def legal_basis_text(results: pd.DataFrame) -> str:
    if results.empty:
        return ""
    labels = []
    for row in results.head(5).to_dict("records"):
        labels.append(f"{row['article_id']} - {row['article']}")
    return "; ".join(labels)


def confidence_label(results: pd.DataFrame) -> str:
    if results.empty:
        return "low"
    top_score = float(results.iloc[0].get("score", 0))
    if top_score >= 25:
        return "high"
    if top_score >= 10:
        return "medium"
    return "low"


def build_kg_trace(
    query: str,
    results: pd.DataFrame,
    data: Any,
) -> dict[str, pd.DataFrame | str]:
    article_ids = _article_ids(results)
    return {
        "article_ids": ", ".join(article_ids),
        "legal_basis": legal_basis_text(results),
        "confidence": confidence_label(results),
        "query_keyphrases": extract_query_keyphrases(query, data.kg_keyphrases),
        "article_keyphrases": keyphrases_for_articles(article_ids, data.article_nodes),
        "concepts": concepts_for_articles(article_ids, data.article_nodes, data.kg_concepts),
        "triples": triples_for_articles(article_ids, data.kg_triples),
        "edges": edges_for_articles(article_ids, data.kg_edges),
        "rules": rules_for_query_and_articles(query, article_ids, data.kg_rules, data.kg_triples),
    }

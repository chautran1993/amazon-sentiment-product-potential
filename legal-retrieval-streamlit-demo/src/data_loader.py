from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\vungo\Downloads\legal_kg_retrieval_comparison_outputs\content\output"
)


@dataclass(frozen=True)
class DemoData:
    output_dir: Path
    corpus: dict[str, dict[str, Any]]
    details: pd.DataFrame
    summary: pd.DataFrame
    eval_dataset: pd.DataFrame
    methods: list[str]
    chart_paths: list[Path]
    article_nodes: dict[str, list[dict[str, Any]]]
    kg_concepts: list[dict[str, Any]]
    kg_relations: list[dict[str, Any]]
    kg_rules: list[dict[str, Any]]
    kg_keyphrases: list[dict[str, Any]]
    kg_triples: pd.DataFrame
    kg_edges: pd.DataFrame
    kb_metadata: dict[str, Any]


def resolve_output_dir(user_value: str | None = None) -> Path:
    raw = user_value or os.getenv("LEGAL_OUTPUT_DIR") or str(DEFAULT_OUTPUT_DIR)
    path = Path(raw).expanduser()
    if path.name.lower() != "output" and (path / "content" / "output").exists():
        path = path / "content" / "output"
    if path.name.lower() != "output" and (path / "output").exists():
        path = path / "output"
    return path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_labels(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def load_demo_data(output_dir: str | Path | None = None) -> DemoData:
    base = resolve_output_dir(str(output_dir) if output_dir else None)
    required_files = [
        "corpus_full.json",
        "retrieval_method_comparison_details.csv",
        "retrieval_method_comparison_summary.csv",
        "retrieval_eval_dataset.csv",
    ]
    missing = [name for name in required_files if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Khong tim thay cac file output bat buoc: "
            + ", ".join(missing)
            + f"\nThu muc dang doc: {base}"
        )

    corpus_items = _read_json(base / "corpus_full.json")
    corpus = {item["id"]: item for item in corpus_items}

    details = pd.read_csv(base / "retrieval_method_comparison_details.csv")
    summary = pd.read_csv(base / "retrieval_method_comparison_summary.csv")
    eval_dataset = pd.read_csv(base / "retrieval_eval_dataset.csv")

    details["gold_list"] = details["gold_labels"].apply(_split_labels)
    details["predicted_list"] = details["predicted_labels"].apply(_split_labels)
    eval_dataset["gold_list"] = eval_dataset["gold_labels"].apply(_split_labels)

    methods = sorted(details["method"].dropna().unique().tolist())
    if "BM25" in methods:
        methods.remove("BM25")
        methods.insert(0, "BM25")
    if "Hybrid_RRF" in methods:
        methods.remove("Hybrid_RRF")
        methods.insert(min(1, len(methods)), "Hybrid_RRF")

    chart_paths = sorted(base.glob("chart_*.png"))

    article_nodes = _read_json(base / "article_nodes.json") if (base / "article_nodes.json").exists() else {}
    kb = _read_json(base / "final_knowledge_base.json") if (base / "final_knowledge_base.json").exists() else {}
    ontology = kb.get("ontology", {}) if isinstance(kb, dict) else {}
    kg_concepts = ontology.get("Conc", [])
    kg_relations = ontology.get("Rel", [])
    kg_rules = ontology.get("Rules", [])
    kg_keyphrases = kb.get("Keyphrases", []) if isinstance(kb, dict) else []
    kb_metadata = kb.get("metadata", {}) if isinstance(kb, dict) else {}

    kg_triples = (
        pd.read_csv(base / "legal_triples_raw.csv")
        if (base / "legal_triples_raw.csv").exists()
        else pd.DataFrame()
    )
    kg_edges = (
        pd.read_csv(base / "final_legal_kg_edges.csv")
        if (base / "final_legal_kg_edges.csv").exists()
        else pd.DataFrame()
    )

    return DemoData(
        base,
        corpus,
        details,
        summary,
        eval_dataset,
        methods,
        chart_paths,
        article_nodes,
        kg_concepts,
        kg_relations,
        kg_rules,
        kg_keyphrases,
        kg_triples,
        kg_edges,
        kb_metadata,
    )


def article_display_name(article_id: str, corpus: dict[str, dict[str, Any]]) -> str:
    item = corpus.get(article_id, {})
    if not item:
        return article_id
    title = item.get("tieu_de") or ""
    law = item.get("van_ban") or ""
    number = item.get("so_dieu") or ""
    prefix = f"{law} - Dieu {number}".strip(" -")
    return f"{prefix}: {title}" if title else prefix


def article_text(article_id: str, corpus: dict[str, dict[str, Any]]) -> str:
    item = corpus.get(article_id, {})
    text = item.get("full_text")
    if text:
        return str(text)
    content = item.get("noi_dung")
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content or "")

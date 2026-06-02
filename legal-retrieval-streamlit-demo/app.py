from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_demo_data  # noqa: E402
from src.kg_trace import build_kg_trace  # noqa: E402
from src.llm_answer import generate_llm_answer  # noqa: E402
from src.retrieval_demo import (  # noqa: E402
    build_comparison_table,
    build_results_table,
    find_closest_question,
    gold_basis_for_question,
    live_bm25_search,
)
from src.sample_qa import find_best_sample_qa, format_sample_answer, load_sample_qa  # noqa: E402


st.set_page_config(page_title="Legal KG RAG Demo", layout="wide")


@st.cache_data(show_spinner=False)
def cached_load(output_dir: str):
    return load_demo_data(output_dir)


@st.cache_data(show_spinner=False)
def cached_load_sample_qa():
    return load_sample_qa()


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def match_quality(score: float) -> str:
    if score >= 0.75:
        return "cao"
    if score >= 0.45:
        return "trung bình"
    return "thấp"


def choose_best_method(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return ""
    ranked = comparison.copy()
    ranked["recall_at_k"] = ranked["recall_at_k"].fillna(0)
    ranked = ranked.sort_values(
        ["hits_in_top_k", "mrr", "recall_at_k", "processing_time_sec"],
        ascending=[False, False, False, True],
    )
    return str(ranked.iloc[0]["method"])


st.title("Truy xuất thông tin Pháp luật Lao động")
st.caption(
    "Nhập câu hỏi pháp luật lao động -> truy xuất điều luật, matching Knowledge Graph, và trả lời kèm căn cứ."
)

with st.sidebar:
    st.header("Cấu hình")
    output_dir = st.text_input(
        "Thư mục output",
        value=r"C:\Users\vungo\Downloads\legal_kg_retrieval_comparison_outputs\content\output",
    )
    st.caption("Có thể trỏ tới thư mục gốc hoặc thư mục `content/output`.")
    st.divider()
    provider = st.selectbox("LLM provider", ["LM Studio", "Gemini", "OpenAI"], index=0)
    has_env_key = bool(
        os.getenv("GEMINI_API_KEY")
        if provider == "Gemini"
        else os.getenv("OPENAI_API_KEY")
        if provider == "OpenAI"
        else True
    )
    use_llm = st.toggle("Dùng LLM để trả lời", value=has_env_key)
    default_model = (
        "google/gemma-4-e2b"
        if provider == "LM Studio"
        else "gemini-2.0-flash"
        if provider == "Gemini"
        else "gpt-4o-mini"
    )
    llm_model = st.text_input("Model", value=default_model)
    lm_base_url = ""
    if provider == "LM Studio":
        lm_base_url = st.text_input(
            "LM Studio base URL",
            value=os.getenv("LM_STUDIO_BASE_URL", "http://192.168.2.54:1234/v1"),
        )
        api_key = ""
    else:
        api_key = st.text_input(
            "GEMINI_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY",
            value="",
            type="password",
            placeholder="Để trống nếu đã set biến môi trường",
        )

try:
    data = cached_load(output_dir)
except Exception as exc:
    st.error(str(exc))
    st.stop()

sample_qa = cached_load_sample_qa()

with st.sidebar:
    method = st.selectbox("Phương pháp so sánh output", ["Tự chọn tốt nhất"] + data.methods)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
    st.divider()
    st.metric("Số điều luật", f"{len(data.corpus):,}")
    st.metric("Concepts", f"{len(data.kg_concepts):,}")
    st.metric("Relations/Triples", f"{len(data.kg_triples):,}")
    st.metric("Rules", f"{len(data.kg_rules):,}")

summary = data.summary.copy()
best_recall = summary.sort_values("Recall@5", ascending=False).iloc[0]
best_mrr = summary.sort_values("MRR", ascending=False).iloc[0]
fastest = summary.sort_values("Avg Processing Time", ascending=True).iloc[0]

tab_qa, tab_samples, tab_about = st.tabs(["Hỏi đáp", "Câu hỏi mẫu", "Giới thiệu"])

with tab_qa:
    query = st.text_area(
        "Nhập câu hỏi pháp luật lao động:",
        value="Một người lao động ký hợp đồng xác định thời hạn 30 tháng với công ty X. Hết hạn 2 tháng (thời hạn thử việc luật định dành cho người lao động này), người lao động vẫn tiếp tục làm việc mà không ký hợp đồng mới. Trong thời gian này, công ty cho người lao động thử việc lại 60 ngày vì “cần đánh giá năng lực mới”. Sau đó công ty chấm dứt vì cho rằng không đạt. Công ty làm như vậy là đúng hay sai?",
        height=110,
        placeholder="Ví dụ: Người lao động thử việc 2 tháng rồi bị chấm dứt có đúng luật không?",
    )
    search = st.button("Truy vấn", type="primary")

    if search or query:
        with st.spinner("Đang truy xuất điều luật và Knowledge Graph..."):
            sample_match = find_best_sample_qa(query, sample_qa)
            retrieval_query = sample_match["question"] if sample_match else query
            live_results = live_bm25_search(retrieval_query, top_k, data.corpus)
            kg_trace = build_kg_trace(retrieval_query, live_results, data)
            if sample_match:
                answer = format_sample_answer(sample_match)
                answer_mode = f"sample_qa_docx_{sample_match['match_score']:.2f}"
            elif use_llm:
                answer, answer_mode = generate_llm_answer(
                    query,
                    live_results,
                    kg_trace,
                    api_key=api_key or None,
                    model=llm_model,
                    provider=provider,
                    base_url=lm_base_url or None,
                )
            else:
                answer, answer_mode = generate_llm_answer(
                    query,
                    live_results,
                    kg_trace,
                    api_key=None,
                    model=llm_model,
                    provider=provider,
                    base_url=lm_base_url or None,
                )

        st.divider()
        st.subheader("Câu trả lời")
        st.caption(f"Độ tin cậy: **{kg_trace['confidence']}** ")
        st.markdown(answer)

        st.subheader(f"Điều luật liên quan ({len(live_results)} điều)")
        if live_results.empty:
            st.warning("Không có kết quả retrieval.")
        else:
            st.dataframe(
                live_results[["rank", "article_id", "score", "method", "article", "content"]],
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Chi tiết xử lý / Knowledge Graph trace", expanded=False):
            st.markdown("**Keyphrases trích từ câu hỏi / KB**")
            qkp = kg_trace["query_keyphrases"]
            if isinstance(qkp, pd.DataFrame) and not qkp.empty:
                st.dataframe(qkp, hide_index=True, use_container_width=True)
            else:
                st.caption("Không có keyphrase khớp trực tiếp trong KB; hệ thống dùng BM25-lite + mở rộng từ đồng nghĩa.")

            st.markdown("**Keyphrases lấy từ article_nodes.json**")
            akp = kg_trace["article_keyphrases"]
            if isinstance(akp, pd.DataFrame) and not akp.empty:
                st.dataframe(akp, hide_index=True, use_container_width=True)
            else:
                st.caption("Không có keyphrase.")

            st.markdown("**Concepts lấy từ ontology/article_nodes.json**")
            concepts = kg_trace["concepts"]
            if isinstance(concepts, pd.DataFrame) and not concepts.empty:
                st.dataframe(concepts, hide_index=True, use_container_width=True)
            else:
                st.caption("Không có concept.")

            st.markdown("**Relations / Triples lấy từ legal_triples_raw.csv**")
            triples = kg_trace["triples"]
            if isinstance(triples, pd.DataFrame) and not triples.empty:
                st.dataframe(triples, hide_index=True, use_container_width=True)
            else:
                st.caption("Không có triple.")

            st.markdown("**KG edges lấy từ final_legal_kg_edges.csv**")
            edges = kg_trace["edges"]
            if isinstance(edges, pd.DataFrame) and not edges.empty:
                st.dataframe(edges, hide_index=True, use_container_width=True)
            else:
                st.caption("Không có edge.")

            st.markdown("**Rules lấy từ ontology Rules**")
            rules = kg_trace["rules"]
            if isinstance(rules, pd.DataFrame) and not rules.empty:
                st.dataframe(rules, hide_index=True, use_container_width=True)
            else:
                st.caption("Không có rule.")

with tab_samples:
    st.subheader("Câu hỏi mẫu")
    for sample in sample_qa:
        with st.expander(f"Câu {sample['id']}: {sample['question'][:100]}...", expanded=False):
            st.markdown(f"**Câu hỏi:** {sample['question']}")

with tab_about:
    st.subheader("Tổng quan thực nghiệm")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Best Recall@5", best_recall["Method"], format_percent(best_recall["Recall@5"]))
    metric_cols[1].metric("Best MRR", best_mrr["Method"], f"{best_mrr['MRR']:.3f}")
    metric_cols[2].metric("Fastest", fastest["Method"], f"{fastest['Avg Processing Time']:.4f}s")

    display_summary = summary.copy()
    for col in display_summary.columns:
        if pd.api.types.is_float_dtype(display_summary[col]):
            display_summary[col] = display_summary[col].round(4)
    st.dataframe(display_summary, hide_index=True, use_container_width=True)

    if data.chart_paths:
        st.subheader("Biểu đồ output có sẵn")
        cols = st.columns(min(4, len(data.chart_paths)))
        for index, chart_path in enumerate(data.chart_paths):
            with cols[index % len(cols)]:
                st.image(str(chart_path), caption=chart_path.name, use_container_width=True)

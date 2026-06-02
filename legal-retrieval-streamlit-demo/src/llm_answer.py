from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd

from .retrieval_demo import generate_reference_answer, normalize_text


def _friendly_llm_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "resource_exhausted" in lowered or "quota" in lowered or "rate-limit" in lowered:
        return (
            "Gemini đang hết quota/rate limit cho model hiện tại. "
            "Hệ thống đã tự fallback sang câu trả lời offline từ Top-K điều luật."
        )
    if "api_key" in lowered or "permission" in lowered or "unauthenticated" in lowered:
        return (
            "API key không hợp lệ hoặc chưa có quyền gọi model. "
            "Hệ thống đã tự fallback sang câu trả lời offline từ Top-K điều luật."
        )
    cleaned = re.sub(r"https?://\S+", "", message)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"Không gọi được LLM ({cleaned[:180]}). Hệ thống đã fallback sang câu trả lời offline."


def _context_from_results(results: pd.DataFrame, max_chars: int = 6500) -> str:
    chunks = []
    for row in results.head(5).to_dict("records"):
        chunks.append(
            "\n".join(
                [
                    f"ARTICLE_ID: {row.get('article_id')}",
                    f"TITLE: {row.get('article')}",
                    f"CONTENT: {row.get('content')}",
                ]
            )
        )
    text = "\n\n---\n\n".join(chunks)
    return text[:max_chars]


def _extract_choice_text(choice: Any) -> str:
    text = getattr(choice, "text", None)
    if text:
        return str(text)

    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return str(content)
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return str(reasoning)

    if hasattr(choice, "model_dump"):
        data = choice.model_dump()
        text = data.get("text")
        if text:
            return str(text)
        message_data = data.get("message") or {}
        for key in ("content", "reasoning_content"):
            value = message_data.get(key)
            if value:
                return str(value)

    return ""


def _answer_is_too_short(text: str) -> bool:
    cleaned = re.sub(r"[*:\s]+", " ", text or "").strip()
    cleaned = re.sub(r"Đánh giá|Kết luận|Giải thích|Căn cứ|Lưu ý", "", cleaned, flags=re.IGNORECASE)
    return len(cleaned.strip()) < 30


def _structured_reference_answer(query: str, results: pd.DataFrame) -> str:
    if results.empty:
        return (
            "Đánh giá: Chưa có điều luật đủ gần để tạo câu trả lời.\n"
            "* Kết luận Cần nhập câu hỏi cụ thể hơn hoặc tăng Top-K.\n"
            "* Giải thích Hệ thống chưa truy xuất được căn cứ phù hợp từ dữ liệu hiện có.\n"
            "* Căn cứ Chưa xác định.\n"
            "* Lưu ý Đây là câu trả lời demo, không thay thế tư vấn pháp lý."
        )

    top_rows = results.head(3).to_dict("records")
    top = top_rows[0]
    basis = "; ".join(str(row.get("article_id", "")) for row in top_rows)
    top_title = str(top.get("article", top.get("article_id", "")))
    content = re.sub(r"\s+", " ", str(top.get("content", ""))).strip()
    explanation = content[:260].rsplit(" ", 1)[0] if len(content) > 260 else content
    if explanation and not explanation.endswith("."):
        explanation += "..."

    query_norm = normalize_text(query)
    asks_pregnancy_termination = (
        ("mang thai" in query_norm or "co bau" in query_norm or "bau" in query_norm)
        and (
            "duoi viec" in query_norm
            or "sa thai" in query_norm
            or "cham dut" in query_norm
            or "don phuong" in query_norm
        )
    )
    if asks_pregnancy_termination:
        return (
            "Đánh giá: Câu hỏi liên quan trực tiếp đến bảo vệ thai sản của người lao động nữ.\n"
            "* Kết luận Công ty không được sa thải hoặc đơn phương chấm dứt hợp đồng lao động vì lý do người lao động đang mang thai.\n"
            "* Giải thích Pháp luật lao động bảo vệ người lao động nữ trong thời gian mang thai, nghỉ thai sản và nuôi con dưới 12 tháng tuổi; cần kiểm tra thêm trường hợp ngoại lệ nếu doanh nghiệp chấm dứt hoạt động hoặc lý do không liên quan đến thai sản.\n"
            f"* Căn cứ {basis}.\n"
            "* Lưu ý Đây là câu trả lời demo dựa trên Top-K điều luật truy xuất."
        )

    return (
        f"Đánh giá: Câu hỏi có căn cứ gần nhất từ {top_title}.\n"
        f"* Kết luận Cần đối chiếu trực tiếp điều luật liên quan nhất: {top_title}.\n"
        f"* Giải thích {explanation or 'Nội dung điều luật được truy xuất nằm trong bảng bên dưới.'}\n"
        f"* Căn cứ {basis}.\n"
        "* Lưu ý Đây là câu trả lời demo dựa trên Top-K điều luật truy xuất."
    )


def _compact_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    marker = "Đánh giá:"
    if marker in text:
        text = marker + text.split(marker, 1)[1]

    label_patterns = {
        "Kết luận": re.compile(r"^\s*[*\-]?\s*Kết luận\s*:?\s*", re.IGNORECASE),
        "Giải thích": re.compile(r"^\s*[*\-]?\s*Giải thích\s*:?\s*", re.IGNORECASE),
        "Căn cứ": re.compile(r"^\s*[*\-]?\s*Căn cứ\s*:?\s*", re.IGNORECASE),
        "Lưu ý": re.compile(r"^\s*[*\-]?\s*Lưu ý\s*:?\s*", re.IGNORECASE),
    }

    assessment_parts: list[str] = []
    bullets: dict[str, str] = {}
    current_label: str | None = None
    seen_assessment = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = "* " + stripped[2:].strip()

        if stripped.startswith(marker):
            seen_assessment = True
            assessment = stripped[len(marker) :].strip()
            if assessment:
                assessment_parts.append(assessment)
            current_label = None
            continue

        matched_label = None
        matched_value = ""
        for label, pattern in label_patterns.items():
            match = pattern.match(stripped)
            if match:
                matched_label = label
                matched_value = pattern.sub("", stripped).strip()
                break

        if matched_label:
            current_label = matched_label
            bullets[matched_label] = matched_value
            continue

        if current_label and current_label in bullets:
            bullets[current_label] = f"{bullets[current_label]} {stripped}".strip()
        elif seen_assessment and not bullets:
            assessment_parts.append(stripped)

    lines = [f"{marker} {' '.join(assessment_parts).strip()}".strip()]
    for label in ["Kết luận", "Giải thích", "Căn cứ", "Lưu ý"]:
        value = bullets.get(label, "").strip()
        if value:
            lines.append(f"* {label} {value}")

    compact = "\n".join(line for line in lines if line.strip()).strip()
    return compact or text


def generate_llm_answer(
    query: str,
    results: pd.DataFrame,
    kg_trace: dict[str, Any],
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    provider: str = "OpenAI",
    base_url: str | None = None,
) -> tuple[str, str]:
    provider_normalized = provider.strip().lower()
    if provider_normalized == "lm studio":
        key = api_key or os.getenv("LM_STUDIO_API_KEY") or "lm-studio"
    elif provider_normalized == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
    else:
        key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return generate_reference_answer(query, results), f"fallback_no_{provider_normalized}_api_key"

    context = _context_from_results(results)
    concepts = kg_trace.get("concepts")
    triples = kg_trace.get("triples")
    concept_text = concepts.head(8).to_string(index=False) if isinstance(concepts, pd.DataFrame) and not concepts.empty else ""
    triple_text = triples.head(8).to_string(index=False) if isinstance(triples, pd.DataFrame) and not triples.empty else ""

    prompt = f"""
Bạn là trợ lý demo tra cứu pháp luật lao động Việt Nam.
Chỉ trả lời dựa trên CONTEXT, CONCEPTS và TRIPLES bên dưới.
Nếu chưa đủ căn cứ, nói rõ là chưa đủ căn cứ.
Trả lời ngắn gọn bằng tiếng Việt, có cấu trúc:
Chỉ xuất phần trả lời cuối cùng, không ghi reasoning/nháp.
Định dạng bắt buộc:
Đánh giá: <1-2 câu đánh giá ngắn>

* Kết luận <một câu>
* Giải thích <một câu>
* Căn cứ <liệt kê điều luật chính>
* Lưu ý <một câu nếu có ngoại lệ/giới hạn>

QUESTION:
{query}

CONTEXT:
{context}

CONCEPTS:
{concept_text}

TRIPLES:
{triple_text}
""".strip()

    if provider_normalized == "gemini":
        try:
            from google import genai

            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text or "", "llm_gemini"
        except Exception as exc:
            fallback = generate_reference_answer(query, results)
            return f"{fallback}\n\n**Ghi chú LLM:** {_friendly_llm_error(exc)}", "fallback_gemini_error"

    try:
        from openai import OpenAI

        if provider_normalized == "lm studio":
            client = OpenAI(
                api_key=key,
                base_url=base_url or os.getenv("LM_STUDIO_BASE_URL") or "http://192.168.2.54:1234/v1",
            )
            text = ""
            try:
                chat_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Tra loi ngan gon bang tieng Viet. Chi dua cau tra loi cuoi cung, khong ghi reasoning.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=700,
                )
                text = _extract_choice_text(chat_response.choices[0])
            except Exception:
                text = ""

            if not text.strip():
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=700,
                )
                text = _extract_choice_text(response.choices[0])
            if not text.strip():
                fallback = generate_reference_answer(query, results)
                return (
                    f"{fallback}\n\n**Ghi chú LLM:** LM Studio trả response rỗng, hệ thống đã fallback sang câu trả lời offline.",
                    "fallback_lm_studio_empty",
                )
            compact = _compact_answer(text)
            if _answer_is_too_short(compact):
                return _structured_reference_answer(query, results), "fallback_lm_studio_short"
            return compact, "llm_lm_studio"
        else:
            client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là trợ lý RAG pháp luật. Không bịa căn cứ ngoài context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        mode = "llm_lm_studio" if provider_normalized == "lm studio" else "llm_openai"
        return _compact_answer(response.choices[0].message.content or ""), mode
    except Exception as exc:
        fallback = generate_reference_answer(query, results)
        return f"{fallback}\n\n**Ghi chú LLM:** {_friendly_llm_error(exc)}", "fallback_openai_error"

# Legal KG Retrieval Streamlit Demo

Demo nhỏ bằng Python/Streamlit để minh họa hệ thống tra cứu pháp luật dựa trên các output thực nghiệm có sẵn của nhóm.

## Dữ liệu sử dụng

Mặc định app đọc dữ liệu tại:

```powershell
C:\Users\vungo\Downloads\legal_kg_retrieval_comparison_outputs\content\output
```

Các file chính được dùng:

- `corpus_full.json`: nội dung điều luật.
- `retrieval_method_comparison_details.csv`: kết quả Top-K theo từng method.
- `retrieval_method_comparison_summary.csv`: bảng metric tổng hợp.
- `retrieval_eval_dataset.csv`: câu hỏi, đáp án và gold labels.
- `chart_*.png`: biểu đồ thực nghiệm nếu có.

App tự nhận diện các method retrieval từ cột `method`, ví dụ `BM25`, `Embedding_PhoBERT_or_ST`, `Hybrid_RRF`, `Star_Graph_Matching`.

## Cài đặt

```powershell
cd "C:\Users\vungo\OneDrive\Documents\New project\legal-retrieval-streamlit-demo"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy local

```powershell
streamlit run app.py
```

Nếu muốn dùng thư mục output khác:

```powershell
$env:LEGAL_OUTPUT_DIR="D:\path\to\content\output"
streamlit run app.py
```

Bạn cũng có thể sửa trực tiếp đường dẫn trong sidebar của giao diện.

## Cách demo hoạt động

Phần **Kết quả tra cứu trực tiếp** chạy BM25-lite ngay trên `corpus_full.json`, nên khi nhập câu hỏi mới app sẽ trả Top-K điều luật theo chính câu hỏi đó.

App có thể dùng LLM để viết câu trả lời theo kiểu RAG nếu bạn bật **Dùng LLM để trả lời** và cung cấp API key trong sidebar hoặc biến môi trường.

- Gemini: dùng `GEMINI_API_KEY`, model mặc định `gemini-2.0-flash`.
- OpenAI: dùng `OPENAI_API_KEY`, model mặc định `gpt-4o-mini`.
- LM Studio: dùng OpenAI-compatible endpoint, mặc định `http://192.168.2.54:1234/v1`, model `google/gemma-4-e2b`.

Nếu không có API key, app tự fallback sang câu trả lời trích xuất offline từ Top-K điều luật.

Phần **Chi tiết xử lý / Knowledge Graph trace** hiển thị:

- Keyphrases khớp từ câu hỏi hoặc KB.
- Keyphrases và concepts lấy từ `article_nodes.json` + ontology.
- Relations/triples lấy từ `legal_triples_raw.csv`.
- KG edges lấy từ `final_legal_kg_edges.csv`.
- Rules lấy từ `final_knowledge_base.json`.

Output so sánh method hiện có chỉ lưu kết quả retrieval theo các câu hỏi trong bộ thực nghiệm, không lưu raw retrieval score cho từng điều luật và cũng không có query embedding cho câu hỏi mới. Vì vậy, phần **so sánh BM25/PhoBERT/Hybrid** sẽ tìm câu hỏi thực nghiệm gần nhất bằng so khớp token/ký tự đơn giản, rồi hiển thị các danh sách Top-K đã có trong file output.

Cột `score` trong bảng tra cứu trực tiếp là BM25-lite score. Cột `score` trong bảng output thực nghiệm là điểm minh họa suy ra từ thứ hạng `1 / rank`, không phải raw score của BM25/PhoBERT/Hybrid. Các metric như MRR, Recall@K, Precision@K và processing time được đọc trực tiếp từ output thực nghiệm.

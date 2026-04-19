# Sentiment Analysis and Product Potential Evaluation on Amazon

Project full-stack phân tích cảm xúc review Amazon và đánh giá tiềm năng sản phẩm. Mục tiêu là kết hợp machine learning, API backend, frontend dashboard và các báo cáo đánh giá để nhóm sinh viên có thể chia việc rõ ràng, dễ bảo trì.

## End-to-End Demo

Huong dan demo day du nam o:

```text
docs/demo_end_to_end.md
```

Chay nhanh pipeline tren laptop:

```powershell
.venv\Scripts\python.exe scripts\run_demo_pipeline.py --input "C:\Users\vungo\Downloads\dataset\electronics_sample.csv" --demo-asin
```

Script tren se load data, tao label, preprocess, train baseline, predict sample review, tao ranking demo neu dataset thieu `asin`, va in lenh chay backend/dashboard.

## Tech Stack

- Backend: FastAPI
- Frontend: ReactJS
- Machine Learning: Python, scikit-learn, Transformers, notebooks/scripts
- Data: raw, processed, artifacts
- Evaluation: metrics, reports, figures
- Dashboard: assets phục vụ trực quan hóa

## Cấu Trúc Thư Mục

```text
.
├── backend/
│   ├── app/
│   │   ├── api/              # Route FastAPI: sentiment, products, evaluation
│   │   ├── core/             # Config, constants, settings, security helpers
│   │   ├── models/           # Code load model ML hoặc ORM models nếu có database
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   └── services/         # Business logic: prediction, scoring, aggregation
│   └── tests/                # Unit/integration tests cho backend
├── frontend/
│   ├── public/               # Static public files
│   └── src/
│       ├── assets/           # Ảnh/icon/font dùng trong React
│       ├── components/       # UI components dùng lại
│       ├── pages/            # Các màn hình chính
│       └── services/         # Hàm gọi API backend
├── ml/
│   ├── configs/              # Config thí nghiệm, model params, dataset params
│   ├── notebooks/            # Notebook EDA, training thử nghiệm, phân tích kết quả
│   ├── scripts/              # Script train, preprocess, evaluate, inference
│   └── tests/                # Test nhỏ cho preprocessing/feature engineering
├── data/
│   ├── raw/                  # Dữ liệu gốc, không chỉnh sửa
│   ├── processed/            # Dữ liệu đã clean/tokenize/split
│   ├── external/             # Dữ liệu bổ sung từ nguồn ngoài
│   └── artifacts/            # Vectorizer, encoder, tokenizer, intermediate outputs
├── models/
│   ├── checkpoints/          # Checkpoint trong quá trình train
│   └── registry/             # Model final đã chọn kèm metadata/version
├── evaluation/
│   ├── figures/              # Confusion matrix, biểu đồ metric, charts
│   ├── metrics/              # File JSON/CSV chứa accuracy, F1, MAE, ranking score
│   └── reports/              # Báo cáo markdown/pdf/html về kết quả đánh giá
├── dashboard/
│   └── assets/
│       ├── images/           # Hình ảnh dùng riêng cho dashboard/report
│       └── styles/           # CSS/theme assets dùng riêng cho dashboard
├── docs/                     # Tài liệu nhóm: API spec, data dictionary, hướng dẫn
├── scripts/                  # Script tiện ích: setup, seed data, run pipeline
├── requirements.txt          # Dependencies cho backend và ML
└── README.md                 # Tổng quan project
```

## Gợi Ý Chia Việc

- Backend team: xây API trong `backend/app/api`, xử lý logic trong `backend/app/services`.
- Frontend team: xây UI trong `frontend/src/pages` và component dùng lại trong `frontend/src/components`.
- ML team: làm EDA trong `ml/notebooks`, đưa code ổn định sang `ml/scripts`.
- Data/evaluation team: quản lý dữ liệu trong `data`, kết quả đánh giá trong `evaluation`.

## Cài Đặt Backend Và ML

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy Backend

Sau khi có file FastAPI entrypoint, ví dụ `backend/app/main.py`:

```bash
uvicorn backend.app.main:app --reload
```

## Quy Ước Dữ Liệu Và Model

- Không commit dữ liệu thật, checkpoint lớn, file sinh tự động hoặc file chứa thông tin nhạy cảm.
- Giữ dữ liệu gốc trong `data/raw` và không chỉnh sửa trực tiếp.
- Mọi bước xử lý dữ liệu nên có script tương ứng trong `ml/scripts`.
- Model final nên được đặt trong `models/registry` kèm mô tả version, metric và ngày train.

## Bước Tiếp Theo Đề Xuất

1. Tạo FastAPI app tối thiểu trong `backend/app/main.py`.
2. Khởi tạo React app trong `frontend`.
3. Thêm notebook EDA đầu tiên trong `ml/notebooks`.
4. Định nghĩa schema dữ liệu review Amazon trong `docs/data_dictionary.md`.

## Làm Sạch Dataset Reviews

Script load và clean review nằm ở `ml/scripts/load_clean_reviews.py`. Script hỗ trợ file CSV hoặc JSON Lines, chỉ giữ các cột cần thiết nếu tồn tại, loại null, duplicate và review quá ngắn.

Ví dụ chạy với file sample:

```bash
python ml/scripts/load_clean_reviews.py --input "C:\Users\vungo\Downloads\dataset\electronics_sample.csv"
```

Ví dụ chạy với file lớn:

```bash
python ml/scripts/load_clean_reviews.py --input "C:\Users\vungo\Downloads\dataset\electronics_small.csv"
```

Output mặc định:

```text
data/processed/clean_reviews.csv
```

## Tao Nhan Sentiment

Script tao cot `sentiment_label` nam o `ml/scripts/create_sentiment_labels.py`.

Quy tac gan nhan:

- Rating 1 hoac 2: `Negative`
- Rating 3: `Neutral`
- Rating 4 hoac 5: `Positive`

Chay script:

```bash
python ml/scripts/create_sentiment_labels.py
```

Output mac dinh:

```text
data/processed/labeled_reviews.csv
evaluation/figures/sentiment_label_distribution.png
```

## EDA Cho Labeled Reviews

Script EDA nam o `ml/scripts/eda_labeled_reviews.py`.

Chay script:

```bash
python ml/scripts/eda_labeled_reviews.py
```

Script se doc:

```text
data/processed/labeled_reviews.csv
```

Va luu cac chart rieng vao:

```text
outputs/eda/
```

Danh sach chart:

- `01_rating_distribution.png`
- `02_sentiment_label_distribution.png`
- `03_top_20_products_by_reviews.png`
- `04_review_length_distribution.png`
- `05_top_words_basic_cleaning.png`
- `06_class_imbalance_percentage.png`

## Preprocessing Text Cho CNN/LSTM

Module preprocessing nam o `ml/scripts/text_preprocessing_cnn_lstm.py`.

Chay truc tiep:

```bash
python ml/scripts/text_preprocessing_cnn_lstm.py
```

Mac dinh script doc:

```text
data/processed/labeled_reviews.csv
```

Va luu artifacts:

```text
artifacts/keras_tokenizer.pkl
artifacts/label_encoder.joblib
```

Module thuc hien lowercase, remove HTML tags, remove URLs, remove special characters, normalize spaces, NLTK tokenize, remove English stopwords, Keras tokenization, padding/truncation `max_len=256`, encode label va chia train/validation/test.

## Baseline TF-IDF + Logistic Regression

Script baseline sentiment 3 lop nam o `ml/scripts/train_tfidf_logreg_baseline.py`.

Chay script:

```bash
python ml/scripts/train_tfidf_logreg_baseline.py
```

Script doc:

```text
data/processed/labeled_reviews.csv
```

Va luu confusion matrix:

```text
evaluation/figures/tfidf_logreg_confusion_matrix.png
```

Baseline nay huu ich vi nhanh, de giai thich va tao moc so sanh truoc khi dung CNN/LSTM.

## CNN Sentiment Model

Script train CNN nam o `ml/scripts/train_cnn_sentiment.py`.

Chay script:

```bash
python ml/scripts/train_cnn_sentiment.py --no-nltk-download
```

Script dung lai preprocessing trong `ml/scripts/text_preprocessing_cnn_lstm.py`, train mo hinh Embedding + Conv1D + GlobalMaxPooling + Dense/Dropout va ho tro `class_weight`.

Model/artifacts duoc luu tai:

```text
models/cnn/best_model.weights.h5
models/cnn/model_config.json
artifacts/keras_tokenizer.pkl
artifacts/label_encoder.joblib
```

Bao cao va hinh anh:

```text
evaluation/figures/cnn_training_loss.png
evaluation/figures/cnn_training_accuracy.png
evaluation/figures/cnn_confusion_matrix.png
evaluation/reports/cnn_metrics.json
evaluation/reports/cnn_classification_report.txt
```

## LSTM/BiLSTM Sentiment Model

Script train LSTM nam o `ml/scripts/train_lstm_sentiment.py`.

Chay script:

```bash
python ml/scripts/train_lstm_sentiment.py --no-nltk-download
```

Mac dinh script dung BiLSTM vi review text co the phu thuoc vao ngu canh ben trai va ben phai cua tu/cum tu. Neu muon dung LSTM mot chieu, them flag:

```bash
python ml/scripts/train_lstm_sentiment.py --no-nltk-download --use-lstm
```

Model/artifacts duoc luu tai:

```text
models/lstm/best_model.weights.h5
models/lstm/model_config.json
artifacts/keras_tokenizer.pkl
artifacts/label_encoder.joblib
```

Charts va bao cao duoc luu tai:

```text
outputs/lstm/lstm_training_loss.png
outputs/lstm/lstm_training_accuracy.png
outputs/lstm/lstm_confusion_matrix.png
outputs/lstm/lstm_metrics.json
outputs/lstm/lstm_classification_report.txt
```

## BERT Fine-Tuning

Script fine-tune `bert-base-uncased` nam o `ml/scripts/train_bert_sentiment.py`.

Khuyen nghi chay tren Google Colab GPU:

```bash
python ml/scripts/train_bert_sentiment.py --epochs 2 --batch-size 16 --max-len 128
```

Test nhanh voi it mau:

```bash
python ml/scripts/train_bert_sentiment.py --sample-size 1000 --epochs 1 --batch-size 16
```

Script doc:

```text
data/processed/labeled_reviews.csv
```

Model va tokenizer duoc luu tai:

```text
models/bert/
```

Ket qua evaluate duoc luu tai:

```text
outputs/bert/bert_confusion_matrix.png
outputs/bert/bert_metrics.json
outputs/bert/bert_classification_report.txt
```

## Product Potential Ranking

Script ranking san pham nam o `ml/scripts/rank_product_potential.py`.

Input can co cac cot:

```text
asin
predicted_sentiment hoac sentiment_label
```

Chay script:

```bash
python ml/scripts/rank_product_potential.py --input path/to/reviews_with_asin_and_sentiment.csv
```

Mac dinh cong thuc:

```text
product_potential_score = 0.7 * normalized_sentiment_mean + 0.3 * normalized_review_volume
```

Co the doi trong so:

```bash
python ml/scripts/rank_product_potential.py --input path/to/reviews.csv --sentiment-weight 0.8 --volume-weight 0.2
```

Output:

```text
outputs/ranking/product_ranking.csv
outputs/ranking/top_20_product_potential.png
```

## Backend FastAPI

Backend nam trong `backend/app/` voi cau truc:

```text
backend/app/main.py
backend/app/routes/
backend/app/services/
backend/app/schemas/
backend/app/core/
```

Chay local:

```bash
uvicorn backend.app.main:app --reload
```

Neu dung virtual environment trong project:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Mo Swagger UI:

```text
http://127.0.0.1:8000/docs
```

API endpoints:

```text
GET  /health
GET  /products
GET  /products/top-ranking
GET  /products/{asin}
POST /predict-review
```

Vi du predict:

```bash
curl -X POST "http://127.0.0.1:8000/predict-review" \
  -H "Content-Type: application/json" \
  -d "{\"review_text\":\"This product is great and works perfectly\"}"
```

Backend doc product data tu:

```text
outputs/ranking/product_ranking.csv
```

Neu file ranking chua co, backend se thu aggregate tu:

```text
data/processed/labeled_reviews.csv
```

Luu y: cac API product can cot `asin`. Dataset sample hien tai khong co `asin`, nen `/products` va `/products/top-ranking` se tra danh sach rong cho den khi co file ranking hoac review CSV co cot `asin`.

## Frontend React Dashboard

Frontend dashboard nam trong `frontend/`.

Chay frontend:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Mo dashboard:

```text
http://127.0.0.1:5173
```

Mac dinh frontend goi backend tai:

```text
http://127.0.0.1:8000
```

Neu muon doi backend URL, tao file `frontend/.env`:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_ENABLE_MOCK_FALLBACK=true
```

Service layer goi API nam o:

```text
frontend/src/services/api.js
frontend/src/services/mockData.js
```

Frontend dang goi cac API chinh:

```text
GET  /overview
GET  /products/top-ranking
GET  /products/{asin}
POST /predict-review
```

Neu backend chua san sang hoac API product tra ve danh sach rong, dashboard se dung mock data fallback de van demo duoc. Dat `VITE_ENABLE_MOCK_FALLBACK=false` neu muon tat mock fallback.

Co the chay backend bang script:

```powershell
scripts\start_backend.cmd
```

Hoac chay truc tiep:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

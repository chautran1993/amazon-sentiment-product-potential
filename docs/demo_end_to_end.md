# End-to-End Demo Guide

Tai lieu nay mo ta flow demo cho project **Amazon Sentiment Analysis and Product Potential Evaluation**.

## Demo Nhanh Tren Laptop

Chay tu thu muc goc project:

```powershell
.venv\Scripts\python.exe scripts\run_demo_pipeline.py --input "C:\Users\vungo\Downloads\dataset\electronics_sample.csv" --demo-asin
```

Flag `--demo-asin` chi dung cho demo dashboard khi dataset sample khong co cot `asin`. Script se tao file:

```text
data/processed/labeled_reviews_with_demo_asin.csv
```

Sau do tinh ranking va luu:

```text
outputs/ranking/product_ranking.csv
```

## Flow Pipeline

### 1. Load du lieu

```powershell
.venv\Scripts\python.exe ml\scripts\load_clean_reviews.py --input "C:\Users\vungo\Downloads\dataset\electronics_sample.csv"
```

Output:

```text
data/processed/clean_reviews.csv
```

### 2. Tao sentiment label

```powershell
.venv\Scripts\python.exe ml\scripts\create_sentiment_labels.py
```

Output:

```text
data/processed/labeled_reviews.csv
evaluation/figures/sentiment_label_distribution.png
```

### 3. Preprocess text

```powershell
.venv\Scripts\python.exe ml\scripts\text_preprocessing_cnn_lstm.py --no-nltk-download
```

Output:

```text
artifacts/keras_tokenizer.pkl
artifacts/label_encoder.joblib
```

### 4. Train hoac load model tot nhat

Baseline nhanh, de demo tren laptop:

```powershell
.venv\Scripts\python.exe ml\scripts\train_tfidf_logreg_baseline.py
```

Output:

```text
models/baseline/tfidf_logreg_model.joblib
evaluation/reports/tfidf_logreg_metrics.json
evaluation/figures/tfidf_logreg_confusion_matrix.png
```

CNN/LSTM co the chay laptop neu co thoi gian:

```powershell
.venv\Scripts\python.exe ml\scripts\train_cnn_sentiment.py --no-nltk-download
.venv\Scripts\python.exe ml\scripts\train_lstm_sentiment.py --no-nltk-download
```

BERT nen chay tren Google Colab GPU:

```text
ml/notebooks/bert_colab_finetune.ipynb
```

### 5. Predict sentiment cho review

Backend se load baseline model tu:

```text
models/baseline/tfidf_logreg_model.joblib
```

Chay backend:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Goi API:

```powershell
curl -X POST "http://127.0.0.1:8000/predict-review" `
  -H "Content-Type: application/json" `
  -d "{\"review_text\":\"This product is great and works perfectly\"}"
```

### 6. Aggregate theo san pham

Can file co cot:

```text
asin
sentiment_label hoac predicted_sentiment
```

Neu dataset da co `asin`:

```powershell
.venv\Scripts\python.exe ml\scripts\rank_product_potential.py --input data\processed\labeled_reviews.csv
```

Neu dataset sample khong co `asin`, dung orchestration demo:

```powershell
.venv\Scripts\python.exe scripts\run_demo_pipeline.py --skip-load --skip-label --skip-preprocess --skip-train --demo-asin
```

### 7. Tinh ranking san pham

Cong thuc:

```text
product_potential_score = 0.7 * normalized_sentiment_mean + 0.3 * normalized_review_volume
```

Output:

```text
outputs/ranking/product_ranking.csv
outputs/ranking/top_20_product_potential.png
```

### 8. Chay backend

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 9. Chay dashboard

```powershell
cd frontend
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Dashboard:

```text
http://127.0.0.1:5173
```

## Demo Tren Google Colab

Khuyen nghi Colab cho BERT:

1. Upload notebook:

```text
ml/notebooks/bert_colab_finetune.ipynb
```

2. Runtime -> Change runtime type -> GPU.
3. Upload:

```text
data/processed/labeled_reviews.csv
```

4. Chay notebook va tai ve:

```text
bert_results.zip
```

## Ghi Chu Khi Demo

- Laptop demo nhanh nen dung TF-IDF + Logistic Regression.
- CNN/LSTM can vai phut tuy may.
- BERT nen chay GPU.
- Dataset sample hien tai khong co `asin`, nen product ranking that can dataset co `asin`; flag `--demo-asin` chi dung de minh hoa dashboard.

# Chay BERT day du tren Google Colab GPU

Tai lieu nay dung de fine-tune `bert-base-uncased` voi nhieu du lieu hon so voi ban smoke test local. Cach nay phu hop cho project sinh vien: van dung script san co trong repo, chay tren GPU Colab, va luu model/metrics ve may.

## 1. Bat GPU tren Colab

Trong Colab:

1. Vao `Runtime`
2. Chon `Change runtime type`
3. Chon `T4 GPU`
4. Bam `Save`

Kiem tra GPU:

```python
!nvidia-smi

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. Cai thu vien can thiet

```python
!pip install -q transformers scikit-learn pandas matplotlib tqdm safetensors accelerate
```

## 3. Upload dataset va script tu may local

Upload 2 file sau:

- `data/processed/labeled_reviews.csv`
- `ml/scripts/train_bert_sentiment.py`

Chay cell nay tren Colab:

```python
from google.colab import files
from pathlib import Path
import shutil

uploaded = files.upload()

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("ml/scripts").mkdir(parents=True, exist_ok=True)

for filename in uploaded:
    if filename.endswith(".csv"):
        shutil.move(filename, "data/processed/labeled_reviews.csv")
    elif filename.endswith(".py"):
        shutil.move(filename, "ml/scripts/train_bert_sentiment.py")

print("Dataset:", Path("data/processed/labeled_reviews.csv").exists())
print("Script:", Path("ml/scripts/train_bert_sentiment.py").exists())
```

## 4. Kiem tra nhanh du lieu

```python
import pandas as pd

df = pd.read_csv("data/processed/labeled_reviews.csv")
print(df.shape)
print(df[["reviewText", "sentiment_label"]].head())
print(df["sentiment_label"].value_counts())
```

## 5. Chay BERT voi cau hinh khuyen nghi

Cau hinh nay chay toan bo dataset hien co, khong dung `--sample-size`.

```python
!python ml/scripts/train_bert_sentiment.py \
  --input data/processed/labeled_reviews.csv \
  --model-name bert-base-uncased \
  --model-dir models/bert \
  --output-dir outputs/bert \
  --max-len 128 \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --test-size 0.15 \
  --val-size 0.15
```

Neu Colab bi out-of-memory, giam batch size:

```python
!python ml/scripts/train_bert_sentiment.py \
  --input data/processed/labeled_reviews.csv \
  --model-name bert-base-uncased \
  --model-dir models/bert \
  --output-dir outputs/bert \
  --max-len 128 \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --weight-decay 0.01
```

Neu muon nhanh hon de demo:

```python
!python ml/scripts/train_bert_sentiment.py \
  --input data/processed/labeled_reviews.csv \
  --model-name bert-base-uncased \
  --model-dir models/bert \
  --output-dir outputs/bert \
  --max-len 128 \
  --epochs 2 \
  --batch-size 16 \
  --sample-size 8000
```

## 6. Xem ket qua

```python
import json
from pathlib import Path

metrics_path = Path("outputs/bert/bert_metrics.json")
report_path = Path("outputs/bert/bert_classification_report.txt")

print(json.dumps(json.loads(metrics_path.read_text()), indent=2))
print(report_path.read_text())
```

Hien thi confusion matrix:

```python
from IPython.display import Image, display

display(Image("outputs/bert/bert_confusion_matrix.png"))
```

## 7. Nen zip model va outputs de tai ve

```python
!zip -r bert_full_results.zip models/bert outputs/bert
from google.colab import files
files.download("bert_full_results.zip")
```

## 8. Cap nhat ket qua vao project local

Sau khi tai `bert_full_results.zip` ve may:

1. Giai nen file zip.
2. Copy thu muc `models/bert/` vao project local.
3. Copy thu muc `outputs/bert/` vao project local.
4. Chay lai bang so sanh model:

```powershell
.venv\Scripts\python.exe ml\scripts\compare_model_results.py
```

Ket qua moi se nam tai:

- `outputs/model_comparison/model_comparison.csv`
- `outputs/model_comparison/macro_f1_comparison.png`
- `outputs/model_comparison/auto_commentary_vi.txt`

## Ghi chu bao cao

Ban local truoc do co the chi la smoke test nen BERT dat diem thap. Khi chay tren Colab GPU voi toan bo dataset va 2-3 epochs, BERT co kha nang hoc ngu canh review tot hon, dac biet voi nhung cau dai va co ngu nghia phuc tap. Tuy nhien, BERT ton chi phi huan luyen cao hon TF-IDF Logistic Regression va CNN, nen can so sanh bang macro F1 cung voi thoi gian train.

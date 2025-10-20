# ============================================================
# Korean Sentiment Classification with Transformers (Trainer)
# - Safe TrainingArguments wrapper (supports older envs)
# - Clean column normalization & text cleaning
# - Valid evaluation + test prediction + submission save
# ============================================================

# !pip install -q "transformers" "datasets" "accelerate" "evaluate" scikit-learn pandas

import os, re, inspect
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -------------------- 경로/설정 --------------------
DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/자연어_대회"
TRAIN = f"{DATA_DIR}/train-2.csv"
VALID = f"{DATA_DIR}/valid.csv"
TEST  = f"{DATA_DIR}/test.csv"
SUB   = f"{DATA_DIR}/submission.csv"

MODEL_NAME = "klue/roberta-base"  # "beomi/KcELECTRA-base-v2022", "monologg/kobert", "xlm-roberta-base" 등 교체하며 실험
RANDOM_SEED = 42
MAX_LEN = 256
NUM_LABELS = 2

set_seed(RANDOM_SEED)

# -------------------- 컬럼 정규화/전처리 --------------------
def normalize_columns(df: pd.DataFrame):
    df.columns = (
        df.columns
          .str.replace("\ufeff", "", regex=False)
          .str.strip()
          .str.lower()
    )
    rename_map = {
        "labels": "label",
        "target": "label",
        "sentiment": "label",
        "document": "text",
        "review": "text",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df

def clean_text(t):
    if not isinstance(t, str):
        t = "" if pd.isna(t) else str(t)
    t = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# -------------------- 데이터 로드 --------------------
train_df = normalize_columns(pd.read_csv(TRAIN))
valid_df = normalize_columns(pd.read_csv(VALID))
test_df  = normalize_columns(pd.read_csv(TEST))

assert "text" in train_df.columns and "label" in train_df.columns, "train에 text/label 필요"
assert "text" in valid_df.columns and "label" in valid_df.columns, "valid에 text/label 필요"
assert "text" in test_df.columns, "test에 text 필요"

# 텍스트 클린
for df in [train_df, valid_df, test_df]:
    df["text"] = df["text"].apply(clean_text)

# 라벨 안전 캐스팅
train_df["label"] = train_df["label"].astype(int)
valid_df["label"] = valid_df["label"].astype(int)

# -------------------- HF Dataset --------------------
train_ds = Dataset.from_pandas(train_df[["text", "label"]])
valid_ds = Dataset.from_pandas(valid_df[["text", "label"]])

# test는 id/text만 유지(대소문자 혼재 대비)
test_keep_cols = [c for c in test_df.columns if c in ["id","Id","ID","text"]]
test_ds = Dataset.from_pandas(test_df[test_keep_cols])

# -------------------- 토크나이저/토크나이즈 --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, padding=False, max_length=MAX_LEN)

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
valid_tok = valid_ds.map(tokenize_fn, batched=True, remove_columns=valid_ds.column_names)
test_tok  = test_ds.map(tokenize_fn,  batched=True, remove_columns=test_ds.column_names)

# 레이블 추가 (remove_columns로 빠졌으니 다시 붙이기)
train_tok = train_tok.add_column("labels", train_df["label"].tolist())
valid_tok = valid_tok.add_column("labels", valid_df["label"].tolist())

collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------- 모델 --------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# -------------------- 메트릭 --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# -------------------- TrainingArguments 호환 래퍼 --------------------
# 현재 환경의 TrainingArguments가 지원하는 키만 자동 선택하여 안전 생성
def make_training_args(
    output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    seed=42,
):
    sig_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    def supports(k): return k in sig_keys

    # 공통(버전 무관)
    cfg = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        seed=seed,
    )
    if supports("fp16"):
        cfg["fp16"] = True

    # 최신 경로: eval/save 전략을 "epoch"로 **항상 동일**하게 맞춤
    if supports("evaluation_strategy") and supports("save_strategy"):
        cfg["evaluation_strategy"] = "epoch"
        cfg["save_strategy"] = "epoch"
        if supports("load_best_model_at_end"):
            cfg["load_best_model_at_end"] = True
        if supports("metric_for_best_model"):
            cfg["metric_for_best_model"] = "f1_macro"
        if supports("report_to"):
            cfg["report_to"] = "none"

    else:
        # 레거시 경로: 전략 키들이 없으면 깔끔히 포기하고 steps 기반으로 대체
        if supports("save_steps"):
            cfg["save_steps"] = 500
        if supports("eval_steps"):
            cfg["eval_steps"] = 500
        # 혹시라도 기본값으로 남아 있을 수 있는 충돌 키 제거
        for k in ("evaluation_strategy","save_strategy","load_best_model_at_end",
                  "metric_for_best_model","report_to"):
            cfg.pop(k, None)

    return TrainingArguments(**cfg)

# -------------------- args 생성 --------------------
args = make_training_args(
    output_dir=f"{DATA_DIR}/checkpoints/{MODEL_NAME.replace('/', '_')}",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    seed=42,
)

# -------------------- Trainer --------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -------------------- Train --------------------
trainer.train()

# -------------------- Evaluate --------------------
metrics = trainer.evaluate()
print("\n[VALID] metrics:", metrics)

# -------------------- Predict & Save Submission --------------------
test_logits = trainer.predict(test_tok).predictions
test_pred = np.argmax(test_logits, axis=-1).astype(int)

# test id 컬럼 안전 추출
id_col = next((c for c in test_df.columns if c.lower() == "id"), None)
if id_col is None:
    test_df["id"] = range(len(test_df))
    id_col = "id"

submission = pd.DataFrame({
    "id": test_df[id_col],
    "label": test_pred
})

os.makedirs(os.path.dirname(SUB), exist_ok=True)
submission.to_csv(SUB, index=False, encoding="utf-8")
print(f"\n✅ submission saved to: {SUB}")
print(submission.head())

# ============================================================
# Korean Sentiment Classification (Single Model: KcELECTRA)
# - One-click train/eval/predict/submission
# - EarlyStopping + Best model restore
# - Few hyperparams at the top
# ============================================================

# !pip install -q "transformers>=4.36,<5.0" "datasets" "accelerate" "evaluate" scikit-learn pandas

import os, re, inspect, json, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# -------------------- 기본 설정 --------------------
DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/자연어_대회"
TRAIN = f"{DATA_DIR}/train-2.csv"
VALID = f"{DATA_DIR}/valid.csv"
TEST  = f"{DATA_DIR}/test.csv"
SUB   = f"{DATA_DIR}/submission.csv"

MODEL_NAME = "beomi/KcELECTRA-base-v2022"  # 단일 모델 고정
RANDOM_SEED = 42
MAX_LEN = 256
NUM_LABELS = 2

# 간단 튠 포인트 (필요시 여기만 바꾸면 됨)
LEARNING_RATE = 2e-5
TRAIN_BATCH  = 32
EVAL_BATCH   = 64
EPOCHS       = 3
USE_CLASS_WEIGHT = False         # 불균형 심하면 True
EARLY_STOP_PATIENCE = 2

set_seed(RANDOM_SEED)

# -------------------- 유틸 --------------------
def normalize_columns(df: pd.DataFrame):
    df.columns = (
        df.columns
          .str.replace("\ufeff", "", regex=False)
          .str.strip()
          .str.lower()
    )
    rename_map = {
        "labels": "label", "target": "label", "sentiment": "label",
        "document": "text", "review": "text",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df

def clean_text(t):
    if not isinstance(t, str):
        t = "" if pd.isna(t) else str(t)
    t = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def detect_precision():
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere+
                return {"bf16": True}
            else:
                return {"fp16": True}
        # MPS/CPU면 정밀도 옵션 생략
        return {}
    except Exception:
        return {}

# -------------------- 데이터 --------------------
train_df = normalize_columns(pd.read_csv(TRAIN))
valid_df = normalize_columns(pd.read_csv(VALID))
test_df  = normalize_columns(pd.read_csv(TEST))

assert "text" in train_df.columns and "label" in train_df.columns, "train에 text/label 필요"
assert "text" in valid_df.columns and "label" in valid_df.columns, "valid에 text/label 필요"
assert "text" in test_df.columns, "test에 text 필요"

for df in [train_df, valid_df, test_df]:
    df["text"] = df["text"].apply(clean_text)

train_df["label"] = train_df["label"].astype(int)
valid_df["label"] = valid_df["label"].astype(int)

train_ds = Dataset.from_pandas(train_df[["text", "label"]])
valid_ds = Dataset.from_pandas(valid_df[["text", "label"]])
test_keep_cols = [c for c in test_df.columns if c in ["id","Id","ID","text"]]
test_ds = Dataset.from_pandas(test_df[test_keep_cols])

# -------------------- 토크나이저/토큰화 --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, padding=False, max_length=MAX_LEN)

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
valid_tok = valid_ds.map(tokenize_fn, batched=True, remove_columns=valid_ds.column_names)
test_tok  = test_ds.map(tokenize_fn,  batched=True, remove_columns=test_ds.column_names)

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

# -------------------- (옵션) 클래스 가중치 Trainer --------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self._loss_fct = nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else nn.CrossEntropyLoss()
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits = outputs.get("logits")
        loss = self._loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# -------------------- TrainingArguments (버전 호환 래퍼) --------------------
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
    # 현재 환경의 TrainingArguments 시그니처를 확인해 존재하는 키에만 값 주입
    sig = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    def has(k): return k in sig

    cfg = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        seed=seed,
        save_total_limit=2,
    )
    # 혼합/저정밀 옵션
    cfg.update(detect_precision())

    # --- 평가/저장 전략 (버전 호환: evaluation_strategy 또는 eval_strategy) ---
    if has("evaluation_strategy"):
        cfg["evaluation_strategy"] = "epoch"
    elif has("eval_strategy"):
        cfg["eval_strategy"] = "epoch"

    if has("save_strategy"):
        cfg["save_strategy"] = "epoch"
    elif has("save_steps"):
        # 아주 오래된 버전 호환: epoch 전략이 없으면 steps로 대체
        cfg["save_steps"] = 500

    # --- 베스트 모델 로드 & 기준 메트릭 설정 ---
    if has("load_best_model_at_end"):
        cfg["load_best_model_at_end"] = True
    if has("metric_for_best_model"):
        cfg["metric_for_best_model"] = "f1_macro"  # compute_metrics의 키와 일치
    if has("greater_is_better"):
        cfg["greater_is_better"] = True           # f1은 클수록 좋음

    # --- 로깅/리포트 ---
    if has("report_to"):
        cfg["report_to"] = "none"

    # 아주 오래된 버전에서는 eval/save가 steps만 있는 경우 EarlyStopping의 요구사항을 만족시키기 위해
    # eval_steps도 함께 지정 (위에서 epoch를 못 넣었을 때만)
    if not (("evaluation_strategy" in cfg) or ("eval_strategy" in cfg)):
        if has("eval_steps"):
            cfg["eval_steps"] = 500

    return TrainingArguments(**cfg)

args = make_training_args(
    output_dir=f"{DATA_DIR}/checkpoints/{MODEL_NAME.replace('/', '_')}",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    seed=RANDOM_SEED,
)

# -------------------- Trainer --------------------
if USE_CLASS_WEIGHT:
    cls_counts = train_df["label"].value_counts().sort_index()
    freqs = cls_counts / cls_counts.sum()
    weights = 1.0 / (freqs + 1e-12)
    weights = torch.tensor(weights.values, dtype=torch.float32)
    TrainerClass = WeightedTrainer
    trainer_kw = {"class_weights": weights}
else:
    TrainerClass = Trainer
    trainer_kw = {}

trainer = TrainerClass(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    **trainer_kw,
)

# -------------------- Train & Evaluate --------------------
trainer.train()
metrics = trainer.evaluate()
print("\n[VALID] metrics:", metrics)

# -------------------- Predict & Save Submission --------------------
test_logits = trainer.predict(test_tok).predictions
test_pred = np.argmax(test_logits, axis=-1).astype(int)

id_col = next((c for c in test_df.columns if c.lower() == "id"), None)
if id_col is None:
    test_df["id"] = range(len(test_df))
    id_col = "id"

submission = pd.DataFrame({"id": test_df[id_col], "label": test_pred})
os.makedirs(os.path.dirname(SUB), exist_ok=True)
submission.to_csv(SUB, index=False, encoding="utf-8")
print(f"\n✅ submission saved to: {SUB}")
print(submission.head())

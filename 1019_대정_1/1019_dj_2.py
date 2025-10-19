import os
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
from google.colab import drive
import torch
from torch.utils.data import Dataset # Dataset 임포트 유지
# EarlyStoppingCallback 및 EvaluationStrategy 관련 임포트 제거
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, BertTokenizer 

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 데이터 로드 및 환경 설정
# ==============================================================================
drive.mount('/content/drive') 

DATA_PATH = "/content/drive/MyDrive/nlp_contest"
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
df_valid = pd.read_csv(os.path.join(DATA_PATH, 'valid.csv')) # 사용되지 않지만 로드
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
print("✅ 데이터 로드 완료")

# D. 모델 정의 및 하이퍼파라미터 설정
MODEL_NAME = "monologg/kobert"  
MAX_LEN = 146            # 🚨 EDA 결과에 따라 146으로 설정
BATCH_SIZE = 32          
NUM_EPOCHS = 3        # 🚨 5 Epoch으로 변경
LEARNING_RATE = 3e-5     

# ==============================================================================
# 2. 전처리 및 특징 추출 (Dataset 및 Tokenizer)
# ==============================================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = tokenizer(
            str(self.texts[idx]), 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LEN, 
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = SentimentDataset(df_train['text'].tolist(), df_train['label'].tolist())
valid_dataset = SentimentDataset(df_valid['text'].tolist(), df_valid['label'].tolist())
test_texts = df_test['text'].tolist()
test_dataset = SentimentDataset(test_texts)
print("✅ Dataset 객체 생성 완료")

# ==============================================================================
# 3. 모델 정의 및 학습 (5 Epoch 단순 반복)
# ==============================================================================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ KoBERT 모델 로드 및 GPU/CPU 할당 완료")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# TrainingArguments 설정 (오류 유발 파라미터 제거)
training_args = TrainingArguments(
    output_dir='./results',                    
    num_train_epochs=NUM_EPOCHS,               # 🚨 5 Epoch 실행
    per_device_train_batch_size=BATCH_SIZE,    
    per_device_eval_batch_size=BATCH_SIZE,     
    warmup_steps=500,                          
    weight_decay=0.01,                         
    logging_dir='./logs',                      
    logging_steps=500,
    learning_rate=LEARNING_RATE,
    # 🚨 오류 유발 파라미터 전부 제거
    report_to="none"                           
)

# D. Trainer 객체 생성 (Callback 및 eval_dataset 제거)
trainer = Trainer(
    model=model,                               
    args=training_args,                        
    train_dataset=train_dataset,               
    # eval_dataset=valid_dataset,             # 검증은 생략하고 학습에만 집중
    compute_metrics=compute_metrics,           
    # callbacks는 제거
)

print("🚀 모델 학습 시작...")
trainer.train()
print("✅ 모델 학습 완료")

# ==============================================================================
# 4. 예측 및 제출 파일 생성 (최종 5 Epoch 모델로 예측)
# ==============================================================================
print("➡️ 테스트 데이터 예측 시작...")
predictions = trainer.predict(test_dataset) 

predicted_labels = np.argmax(predictions.predictions, axis=1)

submission = pd.DataFrame({
    'Id': df_test['id'],
    'label': predicted_labels
})

SUBMISSION_PATH = os.path.join(DATA_PATH, 'submission.csv')
submission.to_csv(SUBMISSION_PATH, index=False)

print("\n🎉 최종 예측 완료 및 submission.csv 파일 생성!")
print("파일 경로: {}".format(SUBMISSION_PATH))
print("제출 형식 확인 (상위 5개):")
print(submission.head())
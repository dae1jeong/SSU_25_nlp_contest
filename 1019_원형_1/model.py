# =====================================================================================
# 1. 라이브러리 및 모듈 임포트
# =====================================================================================
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup # 스케줄러 임포트
from sklearn.metrics import accuracy_score
import time
import sys # sys 모듈 임포트

# =====================================================================================
# 2. 설정 (Configuration)
# =====================================================================================
# [개선 1] klue/bert-base 모델
MODEL_NAME = "klue/bert-base"

# 하이퍼파라미터
MAX_LEN = 128       
BATCH_SIZE = 32     
EPOCHS = 3     # (성능을 보며 4나 5로 늘려보세요)
LEARNING_RATE = 2e-5

# =====================================================================================
# 3. 장치 설정 (GPU or CPU)
# =====================================================================================
# 님의 PC에서 "True"가 나왔으므로, "cuda"가 자동으로 선택됩니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================================================================
# 4. 데이터 로드 및 전처리
# =====================================================================================
print("Loading data...")
try:
    train_df = pd.read_csv('train.csv')
    valid_df = pd.read_csv('valid.csv')
    test_df = pd.read_csv('test.csv')

    # 결측치 처리
    train_df = train_df.dropna(subset=['text'])
    valid_df = valid_df.dropna(subset=['text'])
    test_df['text'] = test_df['text'].fillna('')
    print("Data loading complete.")
except FileNotFoundError as e:
    print(f"오류: {e}")
    print("="*50)
    print("train.csv, valid.csv, test.csv 파일이 스크립트와 같은 폴더에 있는지 확인하세요.")
    print("="*50)
    sys.exit() # 프로그램 중지

# =====================================================================================
# 5. 토크나이저 및 모델 로드
# =====================================================================================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
print("Tokenizer and model loading complete.")

# =====================================================================================
# 6. 데이터셋 클래스 정의
# =====================================================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =====================================================================================
# 7. 데이터셋 및 데이터로더 생성
# =====================================================================================
print("Creating datasets and dataloaders...")
train_dataset = ReviewDataset(
    texts=train_df['text'].values,
    labels=train_df['label'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
valid_dataset = ReviewDataset(
    texts=valid_df['text'].values,
    labels=valid_df['label'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_dataset = ReviewDataset(
    texts=test_df['text'].values,
    labels=[0] * len(test_df),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("Datasets and dataloaders creation complete.")

# =====================================================================================
# 8. 훈련 및 평가 함수 정의
# =====================================================================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# [개선 2] 학습률 스케줄러 적용
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, device, scheduler): # scheduler 추가
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step() # 스케줄러 스텝 추가

        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time
    print(f"  Train Loss: {avg_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s")
    return avg_loss

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"  Validation Accuracy: {accuracy:.4f}")
    return accuracy

# =====================================================================================
# 9. 모델 훈련 실행
# =====================================================================================
print("Starting training...")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_epoch(model, train_loader, optimizer, device, scheduler) # scheduler 전달
    eval_model(model, valid_loader, device)
print("Training complete.")

# =====================================================================================
# 10. 테스트 데이터 예측 및 제출 파일 생성
# =====================================================================================
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            
    return predictions

print("Starting prediction on test data...")
preds = predict(model, test_loader, device)

print("Creating submission file...")
# [개G선 3] Kaggle 제출 형식에 맞게 'Id' (대문자 I)로 컬럼명 수정
submission_df = pd.DataFrame({'Id': test_df['id'], 'label': preds})
submission_df.to_csv('submission.csv', index=False)

print("submission.csv file has been created successfully!")
print("Submission file sample:")
print(submission_df.head())
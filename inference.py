from model import SentimentLSTM
import torch

model_ = SentimentLSTM(vocab_size=30522, embedding_dim=300)
model_.load_state_dict(torch.load("sentiment_lstm.pth", weights_only=False, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

while True:
    text = input('>>> ')
    if text == 'exit':
        break

    pred = model_(text)[0]
    print(f'Model Prediction: {"Positive" if pred > 0.55 else ("Negative" if pred < 0.45 else "Neutral")} ({pred:.4f})')

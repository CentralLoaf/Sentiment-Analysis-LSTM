import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
        
        
# (Using torch.nn) Define the model architecture
class SentimentLSTM(nn.Module):
    def __init__(self, embedding_dim: int = 300):
        
        super().__init__()
        vocab_size = self.prep_data()
        
        # Word embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim)
        
        # Main LSTM layers, 10% dropout
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=256, 
                            num_layers=3, 
                            batch_first=True, 
                            dropout=0.1)
        
        # Define final fully connected layer
        self.dense = nn.Linear(in_features=256, out_features=1, bias=True)
        
        # Using sigmoid for binary classification
        self.activation = nn.Sigmoid()
        
    
    # Load, tokenize and divide the data from the IMDb movie ratings dataset
    def prep_data(self, batch_size: int = 64, sample: bool = False):
        loaded_data = load_dataset('imdb', split={'train': 'train', 'test': 'test'})
        self.vocab_size = 0
        
        # Use BERT tokenizer to split text samples for training
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = tokenizer.vocab_size
        # Define the tokenization of each batch
        self.token_func = lambda sample: tokenizer(sample['text'], padding='max_length', truncation=True, max_length=512)
        # Batch data and apply tokenization function
        self.data = loaded_data.map(self.token_func, batched=True, batch_size=batch_size)
        # Convert dataset to PyTorch-friendly
        self.data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            
        # Differenciate train v test
        train, test = self.data['train'], self.data['test']
        
        # Batchify using DataLoader
        self.train_ = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.test_ = DataLoader(test, batch_size=batch_size, shuffle=True)
            
        if sample:
            print(next(iter(self.train_)))
        return tokenizer.vocab_size
    
    
    # Pass a sentence through the network
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Apply embedding and LSTM layers
        embedded = self.embedding(token_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, attention_mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        
        lstm_out, (hidden, cell) = self.lstm(packed_embedded)
        
        # Final value
        return self.activation(self.dense(hidden[-1])).squeeze(dim=1)
    
    
    def train(self, lr: int = 1e-3, epochs: int = 1000):
        
        # Create LSTM object
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        print(f'Using: {device.title()}')
        
        # Loss func, optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        losses = np.array([np.inf])
        
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            
            for batch in self.train_:
                optimizer.zero_grad()
                
                # Forward / backward passes
                out = self.forward(batch['input_ids'].to(device), 
                                   batch['attention_mask'].to(device))
                loss = criterion(out, batch['label'].to(device).float())
                loss.backward()
                optimizer.step()
                
                # Using training dataset for metric eval for simplicity
                total_loss += loss.item()
                correct += (out > 0.5).sum()
                total += len(out)
            
            np.append(losses, total_loss)
            saved = False
            if losses.min() == total_loss:
                torch.save(self.state_dict(), 'sentiment_lstm.pth')
                saved = True
            print(f'Epoch #{epoch+1} - Loss: {total_loss / len(self.train_)} - Accuracy: {correct / total} - Saved: {saved}')
        
        
    def __call__(self, text: str):
        device = next(self.parameters()).device
        
        # Tokenize the text prior to feeding it into .forward()
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        return super().__call__(tokens['input_ids'], tokens['attention_mask'])
    
    
def main():
    print(f'GPU Available: {torch.cuda.is_available()}\nCUDA Version Installed: {torch.version.cuda}\ncuDNN Installed: {torch.backends.cudnn.version()}\nGPU Cores Available: {torch.cuda.device_count()}\nDevice Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
    model = SentimentLSTM(embedding_dim=300)
    model.prep_data(batch_size=64, sample=False)
    model.train(lr=1e-3, epochs=1000)
   
if __name__ == '__main__':
    main()
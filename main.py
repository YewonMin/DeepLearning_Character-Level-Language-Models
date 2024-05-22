import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)
        
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    batch_size = 64
    epochs = 20
    lr = 0.003
    input_file = 'shakespeare_train.txt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Shakespeare(input_file)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    trn_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=trn_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    
    n_chars = len(dataset.chars)
    hidden_size = 128
    n_layers = 2
    
    rnn_model = CharRNN(n_chars, hidden_size, n_chars, n_layers).to(device)
    lstm_model = CharLSTM(n_chars, hidden_size, n_chars, n_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=lr)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
    
    rnn_train_losses, rnn_val_losses = [], []
    lstm_train_losses, lstm_val_losses = [], []

    best_val_loss = float('inf')  # Initialize with a very high value
    best_rnn_model_path = 'best_rnn_model.pth'
    best_lstm_model_path = 'best_lstm_model.pth'
    
    for epoch in range(epochs):
        rnn_trn_loss = train(rnn_model, trn_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        rnn_train_losses.append(rnn_trn_loss)
        rnn_val_losses.append(rnn_val_loss)
        
        lstm_trn_loss = train(lstm_model, trn_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)
        lstm_train_losses.append(lstm_trn_loss)
        lstm_val_losses.append(lstm_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} RNN - Training Loss: {rnn_trn_loss:.4f}, Validation Loss: {rnn_val_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs} LSTM - Training Loss: {lstm_trn_loss:.4f}, Validation Loss: {lstm_val_loss:.4f}")

        # Check if the current RNN model is the best so far
        if rnn_val_loss < best_val_loss:
            best_val_loss = rnn_val_loss
            torch.save(rnn_model.state_dict(), best_rnn_model_path)
        
        # Check if the current LSTM model is the best so far
        if lstm_val_loss < best_val_loss:
            best_val_loss = lstm_val_loss
            torch.save(lstm_model.state_dict(), best_lstm_model_path)

    # Plot RNN Train Loss
    plt.figure()
    plt.plot(rnn_train_losses, label='RNN Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss')
    plt.legend()
    plt.savefig('rnn_train_loss.png')
    
    # Plot RNN Val Loss
    plt.figure()
    plt.plot(rnn_val_losses, label='RNN Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('RNN Validation Loss')
    plt.legend()
    plt.savefig('rnn_val_loss.png')
    
    # Plot LSTM Train Loss
    plt.figure()
    plt.plot(lstm_train_losses, label='LSTM Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('LSTM Training Loss')
    plt.legend()
    plt.savefig('lstm_train_loss.png')
    
    # Plot LSTM Val Loss
    plt.figure()
    plt.plot(lstm_val_losses, label='LSTM Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('LSTM Validation Loss')
    plt.legend()
    plt.savefig('lstm_val_loss.png')

if __name__ == '__main__':
    main()

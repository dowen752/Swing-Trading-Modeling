import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd

class SwingTradingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        def forward(self, x):
            return self.net(x)
def main():
    
    df = yf.download("MSFT", start="2010-01-01", end="2020-12-31")
    df.reset_index(inplace=True)
    print(df.describe())
    
    X = data
    Y = other_data
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)
    
    model = SwingTradingModel(input_size = X.shape[1])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X.to(device))
        loss = criterion(outputs, Y.to(device))
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    
    
    
if __name__ == "__main__":
    main()
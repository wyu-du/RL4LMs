import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# read data
df = pd.read_excel('data/human_preference/sampled_multidoc2dial.xlsx', index_col=0)
X = torch.tensor(df.iloc[:, 8:12].values, dtype=torch.float)
y = torch.tensor(df.iloc[:, 12].values, dtype=torch.long)

# Define the custom linear neural network
class CustomLinearNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomLinearNet, self).__init__()
        self.coef = nn.parameter.Parameter(torch.ones(1, dtype=torch.float))
        self.linear = nn.Linear(1, num_classes)
    
    def forward(self, x):
        out = torch.max(self.coef * x[:,0] + (1-self.coef) * x[:,1], 
                        self.coef * x[:,2] + (1-self.coef) * x[:,3])
        out = self.linear(out.unsqueeze(1))
        return out

# Set random seed for reproducibility
torch.manual_seed(42)

# Create an instance of the custom linear neural network
model = CustomLinearNet(num_classes=3)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1)
num_epochs = 19

# Training loop
best_acc = 0.
best_epoch = 0
for epoch in range(num_epochs):
    # Forward pass
    output = model(X)
    loss = criterion(output, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss for every epoch
    output = model(X)
    _, predicted_label = torch.max(output, 1)
    acc = (predicted_label == y).float().mean().item()
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Acc: {acc}")

# Test the trained model
print(f'Best_epoch: {best_epoch}, best_acc: {best_acc}')
print('Coef = ', model.coef)
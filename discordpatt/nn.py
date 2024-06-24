import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_pickle('discordpatt/test.pkl')
y = df['3_step_future_change']
X = df.drop(columns = ['3_step_future_change'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)
X_train = torch.tensor(X_train.values,device=device,dtype=torch.float32)
X_test = torch.tensor(X_test.values,device=device,dtype=torch.float32)
y_train = torch.tensor(y_train.values,device=device,dtype=torch.float32)
y_test = torch.tensor(y_test.values,device=device,dtype=torch.float32)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(12,50)
        self.fc2 = nn.Linear(50,30)
        self.fc3 = nn.Linear(30,20)
        self.fc4 = nn.Linear(20,1)
    def forward(self,x):
        x = self.fc4(self.fc3(self.fc2(self.fc1(x))))
        return x
    
model = NN().to(device=device)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loss_values = []
test_loss_values = []
num_epochs = 1000

for epoch in range(num_epochs):
    outputs = model(X_train).squeeze(1)
    loss = criterion(outputs,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    train_loss_values.append(loss.item())

    outputs = model(X_test)
    loss = criterion(outputs,y_test)
    test_loss_values.append(loss.item())


plt.plot(range(1, num_epochs + 1), train_loss_values, marker='o', color='b')
plt.plot(range(1, num_epochs+1), test_loss_values, marker='X', color='k')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()
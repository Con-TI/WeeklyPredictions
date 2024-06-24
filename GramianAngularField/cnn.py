import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_pickle("GramianAngularField/test.pkl")
X = torch.tensor(np.array(df['matrix'].to_list()),device=device, dtype= torch.float32)
y = torch.tensor(np.array(df['y'].to_list()),device=device, dtype= torch.float32)

class ImageDataset(Dataset):
    def __init__(self, data, labels,transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image,label

transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

dataset = ImageDataset(X, y, transform=transform)
train_loader = DataLoader(dataset, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #3 x 15 x 15
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4)
        #16 x 12 x 12
        self.pool = nn.MaxPool2d(2,2)
        #16 x 6 x 6
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        #32 x 4 x 4 
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.view(-1,512)
        x = self.relu((self.fc1(x)))
        x = self.relu((self.fc2(x)))
        x = self.fc3(x).squeeze(0)
        return x

model = CNN().to(device=device)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 100

losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.squeeze(0)
        labels = labels.squeeze(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    average_loss = epoch_loss/len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {average_loss / 10:.3f}')
    losses.append(average_loss)

plt.plot(losses)
plt.show()


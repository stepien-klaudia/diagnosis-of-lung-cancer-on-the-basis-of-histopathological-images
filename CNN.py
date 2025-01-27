import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import functions
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

functions.delete_all_test_jpg_files('pliki_new')
functions.copy_random_data_from_all_folders()

transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = ImageFolder(root = "pliki_new/Training", transform = transform_train)
test_data = ImageFolder(root = "pliki_new/Testing", transform = transform_test)

train_loader = DataLoader(train_data, batch_size=25, shuffle=True)
test_loader = DataLoader(test_data, batch_size=25, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding= 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
net = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0015)

net.train()
for epoch in range(25):
    loss_epoch = 0
    print(f"Epoka {epoch}")
    for i, batch in enumerate(train_loader, 0):
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f"epoch = {epoch}, loss = {loss_epoch}")


correct = 0
total = 0
net.eval()

outputs_all = []
labels_all = []

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        outputs = net(images)
        predicted = torch.argmax(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        outputs_all.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        labels_all.append(labels.numpy())

print(f'ACC: {100 * correct // total}%')
        

outputs_all = np.concatenate(outputs_all)
labels_all = np.concatenate(labels_all)

print(outputs_all)
print(labels_all)

print(outputs_all.shape)
print(labels_all.shape)

y_d = np.argmax(outputs_all, axis=1)

def get_label(x):
    return ['gruczolakorak płuc' if z == 0 else 'łagodna tkanka płuc' if z == 1 else 'rak płaskonabłonkowy płuc' for z in x]

cm = confusion_matrix(y_true=get_label(labels_all), y_pred=get_label(y_d), labels=['gruczolakorak płuc','łagodna tkanka płuc', 'rak płaskonabłonkowy płuc'])
print(classification_report(y_true=get_label(labels_all), y_pred=get_label(y_d), labels=['gruczolakorak płuc','łagodna tkanka płuc', 'rak płaskonabłonkowy płuc']))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['gruczolakorak\n płuc','łagodna\n tkanka\n płuc', 'rak\n płaskonabłonkowy\n płuc'])
disp.plot()
plt.show()
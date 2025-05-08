import os
import torch
import torch.nn as nn
from torch.utils import data
from dataset import myDataset
from model import myNet

data_path = './data/wav/'
save_path = './results/'
os.makedirs(save_path + 'models', exist_ok=True)

batch_size = 64
uttr_len = 300
fre_size = 200
epochs = 100
num_classes = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = myDataset(data_path + 'train', batch_size, uttr_len, fre_size)
valid_set = myDataset(data_path + 'valid', batch_size, uttr_len, fre_size)

train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

model = myNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0.0

for epoch in range(epochs):
    model.train()
    total, correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f'Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}')

    if (epoch + 1) % 5 == 0:
        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{save_path}/models/best_model_epoch{epoch+1}.pt')
            print(f'Saved model at epoch {epoch+1} with val acc {val_acc:.4f}')
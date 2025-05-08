import os
import glob
import torch
from dataset import myDataset
from model import myNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data_path = './data/wav/test'
batch_size = 64
uttr_len = 300
fre_size = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_files = glob.glob('./results/models/best_model_epoch*.pt')
if not model_files:
    raise FileNotFoundError("❌ No model files found in ./results/models/")
model_path = sorted(model_files)[-1]
print(f'✅ Loading model: {model_path}')

test_dataset = myDataset(path=data_path, batch_size=batch_size, uttr_len=uttr_len, fre_size=fre_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = myNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

correct = 0
total = 0
true_labels = []
predicted_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())
test_acc = correct / total
print(f'✅ Test Accuracy: {test_acc:.4f}')
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neutral', 'Happy', 'Sad', 'Angry'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Speech Emotion Recognition")
plt.show()
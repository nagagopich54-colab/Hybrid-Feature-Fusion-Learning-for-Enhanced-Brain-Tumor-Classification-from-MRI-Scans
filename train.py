# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import Hybrid_BEiT_EffNetV2M

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASE = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task"
train_dir = BASE + "/train"
test_dir  = BASE + "/test"

# Config
IMG_SIZE = 224
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15

# Transforms
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Data
full_train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
class_names = full_train_ds.classes
NUM_CLASSES = len(class_names)

train_ds, val_ds = random_split(
    full_train_ds,
    [4000, 1000],
    generator=torch.Generator().manual_seed(42)
)

test_ds = datasets.ImageFolder(test_dir, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model
model = Hybrid_BEiT_EffNetV2M(NUM_CLASSES).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# Train
def train_one_epoch():
    model.train()
    loss_sum, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)

    return loss_sum / len(train_loader), correct / total


# Validate
def eval_val():
    model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

    return correct / total, loss_sum / len(val_loader)


# Loop
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_acc, val_loss = eval_val()
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
          f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

# Test
model.eval()
all_labels, all_preds = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        preds = model(x).argmax(1)
        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.show()

# Report
print(classification_report(all_labels, all_preds, target_names=class_names))
# Test accuracy
test_accuracy = (all_preds == all_labels).mean()
print(f"Test Accuracy : {test_accuracy*100:.2f}%")

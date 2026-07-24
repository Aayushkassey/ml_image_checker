import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os


# 1. Advanced Data Augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25), # Rotation अलि बढाएको
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'ml_waste_system'

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=16, shuffle=True) 
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes 
print(f"Recognized Classes: {class_names}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training Device: {device}")

# 2. Model Architecture with Higher Dropout (0.5)
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
num_ftrs = model.classifier[3].in_features

model.classifier[3] = nn.Sequential(
    nn.Dropout(0.5), # Overfitting रोक्न Dropout 0.5 बनाएको
    nn.Linear(num_ftrs, 4)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Optimizer मा weight_decay थपेको (Overfitting नियन्त्रण गर्न)
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)

# Learning Rate Scheduler: यदि २ epoch सम्म val_loss घटेन भने LR लाई आधा गरिदिन्छ
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# 3. Training Loop
num_epochs = 20
print("\n--- Optimized Training Started ---")

best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 25)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation Loss अनुसार LR घटाउने
        if phase == 'val':
            scheduler.step(epoch_loss)

            # Best Accuracy भएको weights मात्र सेभ गर्ने
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'gatekeeper_model.pth')
                print(f"🌟 New Best Model Saved! Acc: {best_acc:.4f}")

print(f"\n🎯 Highest Validation Accuracy Achieved: {best_acc*100:.2f}%")
print("✅ Best Model saved successfully as gatekeeper_model.pth")
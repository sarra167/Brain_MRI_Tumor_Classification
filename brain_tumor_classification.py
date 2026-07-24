import os
import random
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import kagglehub


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed()


# ─── Data Loading ──────────────────────────────────────────────────────────────

def download_dataset():
    print("Downloading Brain Tumor MRI dataset...")
    cache_path = kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset')
    print(f"Dataset downloaded to: {cache_path}")
    return cache_path


def resolve_data_dir(cache_path):
    if 'Training' in os.listdir(cache_path):
        return cache_path
    for folder in os.listdir(cache_path):
        potential_path = os.path.join(cache_path, folder)
        if os.path.isdir(potential_path) and 'Training' in os.listdir(potential_path):
            return potential_path
    raise FileNotFoundError("Could not find Training directory in dataset")


def create_dataframe_from_directory(directory):
    data = []
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                data.append({
                    'image_path': os.path.join(class_path, img_name),
                    'class': class_name,
                })
    return pd.DataFrame(data)


# ─── Visualization helpers ─────────────────────────────────────────────────────

def display_mri_samples(df, num_samples=4):
    classes = df['class'].unique()
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(num_samples * 3, len(classes) * 3))
    fig.suptitle("MRI samples by class", fontsize=16, y=1.02)

    for i, cls in enumerate(sorted(classes)):
        class_df = df[df['class'] == cls].sample(num_samples, random_state=42)
        for j, (_, row) in enumerate(class_df.iterrows()):
            img = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2:
                axes[i, j].imshow(img, cmap='gray')
            else:
                axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f"Class: {cls}", fontsize=12, loc='left')

    plt.tight_layout()
    plt.show()


def analyze_mri_quality(df, sample_size=200):
    sample_df = df.sample(sample_size, random_state=42)
    contrast, blur_metric = [], []

    for _, row in sample_df.iterrows():
        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            contrast.append(np.std(img))
            # Laplacian variance: higher = sharper
            blur_metric.append(cv2.Laplacian(img, cv2.CV_64F).var())

    return contrast, blur_metric


# ─── Transforms ───────────────────────────────────────────────────────────────

IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Dataset ───────────────────────────────────────────────────────────────────

class BrainMRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.dataframe['class'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        # Read class from the DataFrame column, not from parsing the path
        label_str = self.dataframe.iloc[idx]['class']

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        # ResNet expects 3-channel input; replicate the grayscale channel
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        label = self.class_to_idx[label_str]

        if self.transform:
            image_tensor = self.transform(image_rgb)

        return image_tensor, torch.tensor(label, dtype=torch.long)


# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=4):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    # Replace the head; new parameters have requires_grad=True by default
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# ─── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, num_epochs=15, patience=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
    }

    print("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            all_preds_list = []
            all_labels_list = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds_list.append(preds.cpu())
                all_labels_list.append(labels.cpu())

            all_preds = torch.cat(all_preds_list).numpy()
            all_labels = torch.cat(all_labels_list).numpy()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_f1'].append(epoch_f1)

            print(f"{phase.upper()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print("  -> Validation loss improved, saving model.")
                else:
                    epochs_no_improve += 1
                    print(f"  -> No improvement. Patience: {epochs_no_improve}/{patience}")

        elapsed = time.time() - epoch_start
        print(f"Epoch time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds_list.append(preds.cpu())
            all_labels_list.append(labels.cpu())

    all_preds = torch.cat(all_preds_list).numpy()
    all_labels = torch.cat(all_labels_list).numpy()

    print("Classification Report (Test Set):\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — Test Set', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    return all_preds, all_labels


# ─── Grad-CAM ──────────────────────────────────────────────────────────────────

def denormalize(img_tensor):
    # Constants defined locally so this function works independently of import order
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()
        output.requires_grad_(True)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0][class_idx].backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx


def visualize_gradcam(model, test_dataset, class_names, device, num_samples=5):
    target_layer = model.layer4[-1]
    for param in target_layer.parameters():
        param.requires_grad = True

    grad_cam = GradCAM(model, target_layer)
    sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 5))
    fig.suptitle('Grad-CAM Interpretability on Brain MRI', fontsize=16, y=1.05)

    for i, idx in enumerate(sample_indices):
        img_tensor, label = test_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        heatmap, pred_idx = grad_cam(input_tensor)

        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        original_img = denormalize(img_tensor).permute(1, 2, 0).numpy()
        superimposed = cv2.addWeighted(
            (original_img * 255).astype(np.uint8), 0.6,
            heatmap_colored, 0.4, 0,
        )

        true_class = class_names[label]
        pred_class = class_names[pred_idx]
        color = 'green' if true_class == pred_class else 'red'

        axes[i].imshow(superimposed)
        axes[i].axis('off')
        axes[i].set_title(f"True: {true_class}\nPred: {pred_class}", color=color, fontsize=11)

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/gradcam_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_path = download_dataset()
    DATA_DIR = resolve_data_dir(cache_path)
    TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
    TEST_DIR = os.path.join(DATA_DIR, 'Testing')

    df_train = create_dataframe_from_directory(TRAIN_DIR)
    print(f"Total training images: {len(df_train)}")

    df_train_split, df_val_split = train_test_split(
        df_train, test_size=0.2, random_state=42, stratify=df_train['class']
    )

    train_dataset = BrainMRIDataset(df_train_split, transform=train_transforms)
    val_dataset = BrainMRIDataset(df_val_split, transform=val_test_transforms)

    BATCH_SIZE = 32
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
    )

    model = build_model(num_classes=4).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")

    model, history = train_model(model, train_loader, val_loader, device)

    torch.save(model.state_dict(), 'best_brain_mri_model.pth')
    print("Model saved to best_brain_mri_model.pth")

    df_test = create_dataframe_from_directory(TEST_DIR)
    test_dataset = BrainMRIDataset(df_test, transform=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = sorted(df_test['class'].unique())

    evaluate_model(model, test_loader, device, class_names)
    visualize_gradcam(model, test_dataset, class_names, device)

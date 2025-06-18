import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import torchvision.transforms as transforms

from sklearn.metrics import (
    matthews_corrcoef, f1_score, roc_auc_score, 
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="cv2")


class XRayImageDataset(Dataset):
    """Custom PyTorch Dataset for loading X-Ray images from folders."""
    def __init__(self, image_paths, labels, read_func, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.read_func = read_func
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.read_func(image_path)
        image = np.stack([image]*3, axis=-1)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class ResNetFineTuner:
    """A class to fine-tune a pre-trained ResNet model on the X-Ray dataset."""
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        read_image_func: callable,
        resnet_version: str = "resnet50",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 10
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.read_image_func = read_image_func
        self.resnet_version = resnet_version
        self.batch_size = batch_size
        self.lr = learning_rate
        self.num_epochs = num_epochs
        
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        self.class_names = ["NORMAL", "PNEUMONIA"]
        self.num_classes = len(self.class_names)
        self.model_path = os.path.join(
            self.model_dir,
            f"{self.resnet_version}_best_model.pth"
        )

        self.model = self._create_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._prepare_data()

    def _create_model(self):
        """Loads a pre-trained ResNet, freezes its layers, and replaces the
        final classification layer.
        """
        print(f"Loading pre-trained {self.resnet_version} model...")
        model = getattr(models, self.resnet_version)(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        return model

    def _prepare_data(self, val_split: float = 0.1, test_split: float = 0.1):
        """Loads data and splits it into training, validation, and test sets."""
        print("Preparing data loaders...")
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])

        image_paths = []
        labels = []
        for i, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for ext in ("*.jpeg", "*.png", "*.jpg"):
                paths = glob.glob(os.path.join(class_dir, ext))
                image_paths.extend(paths)
                labels.extend([i] * len(paths))

        print(f"Found {len(image_paths)} total images.")

        full_dataset = XRayImageDataset(
            image_paths=image_paths,
            labels=labels,
            read_func=self.read_image_func,
            transform=data_transforms
        )

        num_data = len(full_dataset)
        num_test = int(num_data * test_split)
        num_val = int(num_data * val_split)
        num_train = num_data - num_test - num_val
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [num_train, num_val, num_test]
        )
        print(
            f"Splitting data into: {len(train_dataset)} "
            f"Train, {len(val_dataset)} "
            f"Validation, {len(test_dataset)} Test samples."
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics for binary classification."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix for sensitivity and specificity
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp  = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            "accuracy": accuracy,
            "mcc": mcc,
            "f1": f1,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": (cm)
        }

    def _evaluate_model(self, data_loader, phase="Validation"):
        """Evaluate model and return comprehensive metrics."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc=f"{phase}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = running_loss / len(data_loader.dataset)
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)

        return avg_loss, metrics

    def train(self):
        """Executes the full training and validation pipeline."""
        best_val_mcc = -1.0

        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")

            # Training Phase
            self.model.train()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in tqdm(self.train_loader, desc="Training"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(
                self.train_loader.dataset
            )
            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Validation Phase
            val_loss, val_metrics = self._evaluate_model(
                self.val_loader, "Validation"
            )

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Metrics:")
            print(f"  Accuracy: {val_metrics["accuracy"]:.4f}")
            print(f"  MCC: {val_metrics["mcc"]:.4f}")
            print(f"  F1-Score: {val_metrics["f1"]:.4f}")
            print(f"  AUC: {val_metrics["auc"]:.4f}")

            # Save model based on MCC
            if val_metrics["mcc"] > best_val_mcc:
                best_val_mcc = val_metrics["mcc"]
                torch.save(self.model.state_dict(), self.model_path)
                print(f"New best model saved with MCC: {best_val_mcc:.4f}")

        print("\nTraining finished.")

    def test_best_model(self):
        """Loads the best performing model and evaluates it on the unseen test
        set with comprehensive metrics."""
        print("\n--- Testing on Unseen Data ---")
        if not os.path.exists(self.model_path):
            print(
                f"Error: Model file not found at {self.model_path}. "
                "Please train the model first."
            )
            return

        # Load the best model weights
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.to(self.device)

        # Evaluate on test set
        test_loss, test_metrics = self._evaluate_model(
            self.test_loader, "Testing"
        )

        # Print comprehensive results
        print("\n" + "="*50)
        print("FINAL PERFORMANCE ON TEST SET")
        print("="*50)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:    {test_metrics["accuracy"]:.4f}")
        print(f"  MCC:         {test_metrics["mcc"]:.4f}")
        print(f"  F1-Score:    {test_metrics["f1"]:.4f}")
        print(f"  ROC-AUC:     {test_metrics["auc"]:.4f}")
        print(f"  Sensitivity: {test_metrics["sensitivity"]:.4f}")
        print(f"  Specificity: {test_metrics["specificity"]:.4f}")

        # Confusion Matrix
        cm = test_metrics["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        cm_disp = ConfusionMatrixDisplay(cm)
        cm_disp.plot(cmap="Blues")
        plt.show()

        print(f"\nDetailed Results:")
        print(f"  Total Test Samples: {len(self.test_loader.dataset)}")
        print(f"  Correct Predictions: {tn + tp}")
        print(f"  Incorrect Predictions: {fp + fn}")

        print("="*50)
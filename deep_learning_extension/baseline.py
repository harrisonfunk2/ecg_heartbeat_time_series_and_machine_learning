import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassPrecision,
                                         MulticlassRecall,
                                         MulticlassF1Score,
                                         MulticlassConfusionMatrix)

import mlflow


def validation_split(dataset, val_size = 0.2, random_seed= 123):
    labels = dataset.labels
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(indices, 
                                                  test_size=val_size, 
                                                  random_state=random_seed, 
                                                  stratify=labels)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset



class ECGDataset(Dataset):
    def __init__(self, csv_path):
        super(ECGDataset, self).__init__()
        df = pd.read_csv(csv_path, header=None)
        features = df.iloc[:, :-1].to_numpy()
        labels = df.iloc[:, -1].to_numpy()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        features = features.unsqueeze(0)
        labels = self.labels[idx]
        return features, labels



    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 46, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



    
def train(model, device, train_loader, val_loader, criterion, optimizer, epochs=25):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0.0, 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        model.eval()
        
        val_loss, val_correct, val_total = 0.0, 0.0, 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')
            print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}')   

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())   

    best_results = {'best_epoch': best_epoch,
                    'best_val_acc':  best_val_acc,
                    'best_val_loss': best_val_loss}  
        
    print(f'\nBest Epoch: {best_epoch}'
            f'\nBest Validation Accuracy: {best_val_acc}'
            f'\nBest Validation Loss: {best_val_loss}')


    return history, best_results, best_model_state




def evaluate(
        model, device, dataloader, accuracy_metric, 
        precision_metric, recall_metric, f1_metric, 
        confusion_matrix_metric
        ):
    model.eval()
    
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    confusion_matrix_metric.reset()
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)

            accuracy_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            confusion_matrix_metric.update(preds, labels)

    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    confusion_matrix = confusion_matrix_metric.compute().cpu().numpy()

    results = {"accuracy": accuracy.item(),
               "precision": precision.item(),
               "recall": recall.item(),
               "f1": f1.item(),
               "confusion_matrix": confusion_matrix
               }

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 score: {f1:.4f}')
    # print(f'Confusion Matrix:\n{confusion_matrix}')
    return results






def plot_confusion_matrix(confusion_matrix):
    display_labels = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, 
                                  display_labels=display_labels)

    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Confusion Matrix")

    fig.tight_layout()

    return fig




def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Loss plot
    axes[0].plot(
        epochs,
        history['train_loss'],
        color='blue',
        label='Train Loss'
    )
    axes[0].plot(
        epochs,
        history['val_loss'],
        color='red',
        label='Validation Loss'
    )

    axes[0].set_title('Model Loss (Cross-Entropy)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(
        epochs,
        history['train_acc'],
        color='blue',
        label='Train Accuracy'
    )
    axes[1].plot(
        epochs,
        history['val_acc'],
        color='red',
        label='Validation Accuracy'
    )

    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig



def main(plot=True, test_model=False):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    train_data = ECGDataset('data/mitbih_train.csv') 
    test_data = ECGDataset('data/mitbih_test.csv')

    train_dataset, validation_dataset = validation_split(train_data, 
                                                         val_size=0.2, 
                                                         random_seed=123)

    # Dataloader:
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Evaluation metrics:
    accuracy_metric = MulticlassAccuracy(num_classes=5, average='micro').to(device)
    precision_metric = MulticlassPrecision(num_classes=5, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=5, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=5, average='macro').to(device)
    confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=5).to(device)



    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ECG Multiclass CNN")

    model = CNN().to(device)

    num_epochs = 25
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(run_name = 'Base model'):
        mlflow.log_params({
            'num_epochs': num_epochs,
            'lr': lr,
            'batch_size': batch_size,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss',
            'validation_size': 0.2,
            'conv_channels': '16-32-64',
            'kernel_size': '3-3-3',
            'stride': '1-1-1',
            'padding': '1-1-1',
            'pooling_kernel_size': '2-2',
            'dropout_rate': 0.0
            })

        mlflow.log_text(str(model), "model_architecture.txt")

        # Training:
        history, best_results, best_model_state = train(model=model, 
                                    device=device, 
                                    train_loader = train_loader, 
                                    val_loader = val_loader, 
                                    criterion = criterion, 
                                    optimizer = optimizer, 
                                    epochs=num_epochs)

        if best_model_state is None:
            raise ValueError("Best model state is None. Training may have failed.")
        
        model.load_state_dict(best_model_state)
        results_val = evaluate(model=model, 
                                        device=device, 
                                        dataloader=val_loader, 
                                        accuracy_metric=accuracy_metric,
                                        precision_metric=precision_metric,
                                        recall_metric=recall_metric,
                                        f1_metric=f1_metric,
                                        confusion_matrix_metric=confusion_matrix_metric)

        mlflow.log_metrics({
            'best_epoch': best_results['best_epoch'],
            'best_val_acc': best_results['best_val_acc'],
            'best_val_loss': best_results['best_val_loss'],
            'val_precision': results_val['precision'],
            'val_recall': results_val['recall'],
            'val_f1': results_val['f1']
            })

        for epoch in range(len(history['train_loss'])):
            mlflow.log_metrics({
                'train_loss': history['train_loss'][epoch],
                'train_acc': history['train_acc'][epoch],
                'val_loss': history['val_loss'][epoch],
                'val_acc': history['val_acc'][epoch]
            }, step = epoch + 1)
   
        history_fig = plot_training_history(history)
        confusion_fig = plot_confusion_matrix(results_val['confusion_matrix'])
        
        mlflow.log_figure(history_fig, 'training_history.png')
        mlflow.log_figure(confusion_fig, 'confusion_matrix.png')

        if plot:
            plt.show()

    # Testing:
    if test_model:
        results_test = evaluate(model=model, 
                                         device=device, 
                                         dataloader=test_loader, 
                                         accuracy_metric=accuracy_metric,
                                         precision_metric=precision_metric,
                                         recall_metric=recall_metric,
                                         f1_metric=f1_metric,
                                         confusion_matrix_metric=confusion_matrix_metric)
                    
        if plot:
            plt.show()


    return 
    

    
if __name__ == '__main__':
    main()
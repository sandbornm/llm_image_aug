

from comet_ml.integration.pytorch import log_model
from comet_ml import Experiment

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import json
import os

from data import get_train_loader, get_val_test_loaders
from model import get_model


experiment = Experiment(
  api_key=os.environ["COMET_APIK"],
  project_name="llm-image-augmentation",
  workspace="sandbornm",
  log_graph=True,
  auto_metric_logging=True
)

def forward_pass(model, loader, device, criterion=None, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if criterion:
            loss = criterion(outputs, labels)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader) if criterion else None
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def train_model(model, epochs, train_loader, val_loader, optimizer, criterion, device, model_path, performance_path, save=True):
    model.to(device)
    performance_history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        train_loss, train_accuracy = forward_pass(model, train_loader, device, criterion, optimizer)
        val_loss, val_accuracy = forward_pass(model, val_loader, device, criterion)
        
        performance_history['train_loss'].append(train_loss)
        performance_history['train_accuracy'].append(train_accuracy)
        performance_history['val_loss'].append(val_loss)
        performance_history['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
    if save:
        torch.save(model.state_dict(), model_path)
        with open(performance_path, 'w') as f:
            json.dump(performance_history, f)
        
    return model_path

def test_model(model, test_loader, device):
    model.to(device)
    accuracy = forward_pass(model, test_loader, device)[1]  # No need for criterion or optimizer
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('model_name', type=str, help='Name of the model to use')
    parser.add_argument('train_dir', type=str, help='path to the dataset to train on')
    parser.add_argument('test_dir', type=str, help='path to the dataset to test on')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--weights', type=str, default='default', help='which weights to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    args = parser.parse_args()
    model_name = args.model_name
    epochs = args.epochs
    weights = args.weights
    train_dir = args.train_dir
    batch_size = args.batch_size
    lr = args.lr

    train_loader = get_train_loader(train_dir, batch_size=batch_size)
    val_loader, test_loader = get_val_test_loaders(args.test_dir, batch_size=batch_size)
    num_classes = len(train_loader.dataset.classes)
    
    train_data_name = train_dir.split('/')[-2]
    print(f"Training model {model_name} on {train_data_name} dataset for {epochs} epochs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = get_model(model_name, num_classes, weights=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_path = os.path.join(os.getcwd(), 'models', f'{model_name}_{weights}_{train_data_name}.pth')
    performance_path = os.path.join(os.getcwd(), 'data', 'model_performance', f'{model_name}_{weights}_{train_data_name}.json')

    print(f"training model {model_name} on {train_data_name} dataset for {epochs} epochs...")
    
    hparams = {
        'model_name': model_name,
        'train_data': train_data_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weights': weights,
        'criterion': 'CrossEntropyLoss',
    }

    experiment.log_parameters(hparams)

    trained_model_path = train_model(model, epochs, train_loader, val_loader, optimizer, criterion, device, model_path, performance_path)
    
    exp_name = f"{model_name}_{weights}_{train_data_name}"

    print(f"model type: {type(model)}")
    experiment.log_model(exp_name, model_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Emptied CUDA cache")
    
    test_acc = test_model(model, test_loader, device)
    print(f"Model: {model_name} trained for {epochs} epochs, Test Accuracy: {test_acc:.2f}%")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.spectrogram_dataset import SpectrogramDataset
from models.cnn import CNN
from utils import save_model
import yaml

def train_model(config):
    # Load dataset
    train_dataset = SpectrogramDataset(config['data']['train_spectrograms'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize model, loss function, and optimizer
    model = CNN(num_classes=config['data']['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], Loss: {running_loss/len(train_loader):.4f}')

    # Save the trained model
    save_model(model, config['training']['model_save_path'])

if __name__ == "__main__":
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_model(config)
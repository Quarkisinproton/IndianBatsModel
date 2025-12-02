import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.spectrogram_dataset import SpectrogramDataset
from datasets.spectrogram_with_features_dataset import SpectrogramWithFeaturesDataset
from models.cnn import CNN
from models.cnn_with_features import CNNWithFeatures
from utils import save_model
import yaml
import numpy as np
import json

def train_model(config):
    # Resolve config values with fallbacks
    data_cfg = config.get('data', {})
    train_dir = data_cfg.get('train_spectrograms') or 'data/processed/spectrograms'
    num_classes = data_cfg.get('num_classes') or config.get('model', {}).get('num_classes') or 3
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size') or 16
    lr = train_cfg.get('learning_rate') or 1e-3

    # If features CSV exists, use fused dataset
    features_csv = data_cfg.get('features_csv')
    if features_csv and os.path.exists(features_csv):
        train_dataset = SpectrogramWithFeaturesDataset(train_dir, features_csv)
        use_features = True
    else:
        train_dataset = SpectrogramDataset(train_dir, transform=None)
        use_features = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if use_features:
        # infer numeric feature dimension from a sample
        sample_image, sample_feat, sample_label = train_dataset[0]
        feat_dim = sample_feat.numel()
        model = CNNWithFeatures(num_classes=num_classes, numeric_feat_dim=feat_dim, pretrained=False).to(device)
    else:
        model = CNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = train_cfg.get('num_epochs') or 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if use_features:
                inputs, feats, labels = batch
                inputs = inputs.to(device)
                feats = feats.to(device)
                labels = labels.to(device)
                outputs = model(inputs, feats)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        if num_batches:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/num_batches:.4f}')
        else:
            print('No training data found. Check processed spectrograms path:', train_dir)
            break

    # Save the trained model
    model_save_path = train_cfg.get('model_save_path') or 'models/bat_model.pth'
    os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
    save_model(model, model_save_path)
    print('Saved model to', model_save_path)

if __name__ == "__main__":
    # Load configuration
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_model(config)
import torch
from torch.utils.data import DataLoader
from datasets.spectrogram_dataset import SpectrogramDataset
from models.cnn import CNN
from utils import load_model

def evaluate_model(model_path, data_path, batch_size=16):
    # Load the trained model
    model = load_model(model_path)
    model.eval()

    # Prepare the dataset and dataloader
    dataset = SpectrogramDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}')

if __name__ == "__main__":
    model_path = 'path/to/your/trained/model.pth'
    data_path = 'data/processed/spectrograms'
    evaluate_model(model_path, data_path)
def load_model(model_path):
    import torch
    model = torch.load(model_path)
    model.eval()
    return model

def save_model(model, model_path):
    import torch
    torch.save(model, model_path)

def load_data(data_path):
    import os
    from torchvision import transforms
    from PIL import Image

    images = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = Image.open(img_path)
                images.append(image)
                labels.append(label)

    return images, labels

def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
"""ResNet18 architecture and MRI preprocessing used for NeuroMemory."""
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

CLASS_NAMES = ["Non Demented", "Very mild Dementia", "Mild Dementia"]

def build_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model

def image_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def load_model(checkpoint_path, device="cpu"):
    model = build_model()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

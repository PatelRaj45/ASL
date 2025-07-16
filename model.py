
import torch
import torchvision

from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seeds(seed: int=42):
    """Sets random seeds for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def create_resnet_model(num_classes: int = 29,
                        seed: int = 42,
                        device = device):

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    model = torchvision.models.resnet50(weights=weights).to(device)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    set_seeds()
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048,
                  out_features=num_classes)
    ).to(device)

    return model, transform

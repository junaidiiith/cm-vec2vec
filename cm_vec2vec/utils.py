import torch

def get_device():
    """Get the device to use for training and evaluation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
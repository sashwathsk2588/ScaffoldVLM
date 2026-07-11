"""
Device utility functions to handle CUDA availability gracefully.
"""
import torch

def get_device():
    """
    Get the appropriate device (CUDA if available, otherwise CPU).
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_device_with_index(index=0):
    """
    Get the appropriate device with index (CUDA if available, otherwise CPU).
    
    Args:
        index (int): CUDA device index (ignored if CUDA not available)
        
    Returns:
        str: Device string ('cuda:0' or 'cpu')
    """
    if torch.cuda.is_available():
        return f"cuda:{index}"
    else:
        return "cpu"

def to_device(tensor, device=None):
    """
    Move tensor to the specified device, with fallback to CPU if CUDA not available.
    
    Args:
        tensor: PyTorch tensor
        device: Target device (if None, uses get_device())
        
    Returns:
        torch.Tensor: Tensor on the target device
    """
    if device is None:
        device = get_device()
    
    # If requesting CUDA but it's not available, fallback to CPU
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    
    return tensor.to(device)

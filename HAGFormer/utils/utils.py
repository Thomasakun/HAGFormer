import torch
import numpy as np
import random

def set_random_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, optimizer, epoch, accuracy, path):
    """
    Save the model and optimizer state.

    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        accuracy (float): The validation accuracy.
        path (str): The path to save the model.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(state, path)

def load_model(path, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint.

    Args:
        path (str): The path to the checkpoint.
        model (nn.Module): The model to load the state into.
        optimizer (optim.Optimizer): The optimizer to load the state into.

    Returns:
        Tuple[int, float]: The epoch to resume from and the validation accuracy at the checkpoint.
    """
    state = torch.load(path)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['epoch'], state['accuracy']

import torch
import argparse
import yaml
import os
import logging
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # 添加导入
from training.train import train_model
from data.processing import preprocess_data
from utils.utils import set_random_seed

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def check_params(params):
    """
    Check the validity of the hyperparameters.

    Args:
        params (dict): Dictionary containing hyperparameters.

    Raises:
        AssertionError: If any hyperparameter is invalid.
    """
    try:
        assert params['num_epochs'] > 0, "num_epochs must be positive"
        assert params['batch_size'] > 0, "batch_size must be positive"
        assert params['learning_rate'] > 0, "learning_rate must be positive"
        assert params['input_dim'] > 0, "input_dim must be positive"
        assert params['num_classes'] > 0, "num_classes must be positive"
        assert params['dim_feedforward'] > 0, "dim_feedforward must be positive"
        assert params['nhead'] > 0, "nhead must be positive"
        assert params['num_layers'] > 0, "num_layers must be positive"
        assert 0 <= params['dropout'] < 1, "dropout must be in the range [0, 1)"
    except AssertionError as e:
        logging.error(f"Parameter check failed: {e}")
        raise

parser = argparse.ArgumentParser(description='Training a Transformer Model')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
args = parser.parse_args()
params = load_config(args.config)
check_params(params)
set_random_seed(42)  # Set random seed

# Set default value for 'resume' if not present
if 'resume' not in params:
    params['resume'] = None

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/{current_time}'
tensorboard_dir = f'tensorboard/{current_time}'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, format='%(asctime)s %(message)s')
writer = SummaryWriter(log_dir=tensorboard_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

# Load and preprocess data
S2Feature_train, TrainLabel_train, S2Feature_val, TrainLabel_val = preprocess_data(
    params['train_data_path'], params['train_data_path'], params['val_data_path'], params['val_data_path'], device
)

train_dataset = TensorDataset(S2Feature_train, TrainLabel_train)
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

val_dataset = TensorDataset(S2Feature_val, TrainLabel_val)
val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

# Train the model
train_model(params, device, train_loader, val_loader, writer)

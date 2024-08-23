import os
import h5py
import logging
import torch
from sklearn.preprocessing import StandardScaler

def check_file(file_path):
    """
    Check if a file exists.

    Args:
        file_path (str): Path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        logging.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")

def load_features(file_path):
    """
    Load features from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        np.ndarray: Loaded features.

    Raises:
        Exception: If loading the features fails.
    """
    check_file(file_path)
    try:
        with h5py.File(file_path, 'r') as file:
            t3_feature = file['T3Feature'][:]
            return t3_feature.T
    except Exception as e:
        logging.error(f"Failed to load features from {file_path}: {e}")
        raise

def load_labels(file_path):
    """
    Load labels from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        np.ndarray: Loaded labels.

    Raises:
        Exception: If loading the labels fails.
    """
    check_file(file_path)
    try:
        with h5py.File(file_path, 'r') as file:
            labels = file['TrainLabel'][:]
            return labels.squeeze() - 1
    except Exception as e:
        logging.error(f"Failed to load labels from {file_path}: {e}")
        raise

def preprocess_data(train_features_path, train_labels_path, val_features_path, val_labels_path, device):
    """
    Preprocess the training and validation data.

    Args:
        train_features_path (str): Path to the training features file.
        train_labels_path (str): Path to the training labels file.
        val_features_path (str): Path to the validation features file.
        val_labels_path (str): Path to the validation labels file.
        device (torch.device): Device to load data onto.

    Returns:
        tuple: Processed training and validation features and labels.
    """
    S2Feature_train = load_features(train_features_path)
    TrainLabel_train = load_labels(train_labels_path)
    S2Feature_val = load_features(val_features_path)
    TrainLabel_val = load_labels(val_labels_path)

    scaler = StandardScaler()
    S2Feature_train = scaler.fit_transform(S2Feature_train)
    S2Feature_val = scaler.transform(S2Feature_val)

    S2Feature_train = torch.tensor(S2Feature_train, dtype=torch.float64).to(device)
    S2Feature_val = torch.tensor(S2Feature_val, dtype=torch.float64).to(device)
    TrainLabel_train = torch.tensor(TrainLabel_train, dtype=torch.long).to(device)
    TrainLabel_val = torch.tensor(TrainLabel_val, dtype=torch.long).to(device)

    return S2Feature_train, TrainLabel_train, S2Feature_val, TrainLabel_val

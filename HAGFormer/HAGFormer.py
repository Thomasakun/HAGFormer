import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from datetime import datetime
from tqdm import tqdm
import argparse
import yaml
import random
from typing import Optional, Tuple, Union, Callable

# 设置随机种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置文件加载
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 参数检查
def check_params(params):
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

# 文件检查
def check_file(file_path):
    if not os.path.isfile(file_path):
        logging.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")

# 配置和日志设置
parser = argparse.ArgumentParser(description='Training a Transformer Model')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
args = parser.parse_args()
params = load_config(args.config)
check_params(params)
set_random_seed(42)  # 设置随机种子

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/{current_time}'
tensorboard_dir = f'tensorboard/{current_time}'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, format='%(asctime)s %(message)s')
writer = SummaryWriter(log_dir=tensorboard_dir)

# 确保使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

# 加载和预处理数据
def load_features(file_path):
    check_file(file_path)
    try:
        with h5py.File(file_path, 'r') as file:
            t3_feature = file['T3Feature'][:]
            return t3_feature.T
    except Exception as e:
        logging.error(f"Failed to load features from {file_path}: {e}")
        raise

def load_labels(file_path):
    check_file(file_path)
    try:
        with h5py.File(file_path, 'r') as file:
            labels = file['TrainLabel'][:]
            return labels.squeeze() - 1
    except Exception as e:
        logging.error(f"Failed to load labels from {file_path}: {e}")
        raise

S2Feature_train = load_features(params['train_data_path'])
TrainLabel_train = load_labels(params['train_data_path'])

S2Feature_val = load_features(params['val_data_path'])
TrainLabel_val = load_labels(params['val_data_path'])

scaler = StandardScaler()
S2Feature_train = scaler.fit_transform(S2Feature_train)
S2Feature_val = scaler.transform(S2Feature_val)

S2Feature_train = torch.tensor(S2Feature_train, dtype=torch.float64).to(device)
S2Feature_val = torch.tensor(S2Feature_val, dtype=torch.float64).to(device)
TrainLabel_train = torch.tensor(TrainLabel_train, dtype=torch.long).to(device)
TrainLabel_val = torch.tensor(TrainLabel_val, dtype=torch.long).to(device)

train_dataset = TensorDataset(S2Feature_train, TrainLabel_train)
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

val_dataset = TensorDataset(S2Feature_val, TrainLabel_val)
val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V)

        return weighted_values

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False):
        super(CustomGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, h=None):
        return self.gru(x, h)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dim_feedforward=512, nhead=3, num_layers=2, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.self_attention = SelfAttention(input_dim)
        self.gru = CustomGRU(input_dim, input_dim, batch_first=True)
        encoder_layer = CustomTransformerEncoderLayer(d_model=2*input_dim, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(2*input_dim, num_classes)

    def forward(self, src):
        attention_output = self.self_attention(src)
        gru_output, _ = self.gru(src)
        concat_output = torch.cat((attention_output, gru_output), dim=-1)
        transformed = self.transformer_encoder(concat_output)
        output = self.classifier(transformed)
        return output

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        n_classes = input.size(-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

def save_model(model, optimizer, epoch, accuracy, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(state, path)

def load_model(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['epoch'], state['accuracy']

model = TransformerClassifier(input_dim=params['input_dim'], num_classes=params['num_classes'],
                              dim_feedforward=params['dim_feedforward'], nhead=params['nhead'],
                              num_layers=params['num_layers'], dropout=params['dropout']).to(device).to(torch.float64)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

start_epoch = 0
best_accuracy = 0.0

# 如果指定了检查点路径，则加载检查点
if args.resume:
    try:
        start_epoch, best_accuracy = load_model(args.resume, model, optimizer)
        logging.info(f'Resumed from checkpoint {args.resume}, starting at epoch {start_epoch} with best accuracy {best_accuracy:.4f}')
    except Exception as e:
        logging.error(f'Failed to load checkpoint {args.resume}: {e}')
        raise

for epoch in range(start_epoch, params['num_epochs']):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    epoch_start_time = datetime.now()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{params["num_epochs"]}', unit='batch') as pbar:
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += y_batch.size(0)
            correct_predictions += (predicted == y_batch).sum().item()

            pbar.set_postfix({'loss': f'{total_loss / (total_predictions / params["batch_size"]):.4f}', 'accuracy': f'{100 * correct_predictions / total_predictions:.2f}%'})
            pbar.update(1)

    train_accuracy = 100 * correct_predictions / total_predictions

    logging.info(f'Epoch {epoch + 1}/{params["num_epochs"]} started at {epoch_start_time} and ended at {datetime.now()}')
    logging.info(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

    model.eval()
    correct, total = 0, 0
    all_predicted, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            all_predicted.extend(predicted.view(-1).tolist())
            all_labels.extend(y_batch.view(-1).tolist())

    accuracy = correct / total
    kappa = cohen_kappa_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted, average='weighted')
    recall = recall_score(all_labels, all_predicted, average='weighted')
    f1 = f1_score(all_labels, all_predicted, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predicted)

    logging.info(f'Validation Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')

    writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    writer.add_scalar('Kappa/val', kappa, epoch)
    writer.add_scalar('Precision/val', precision, epoch)
    writer.add_scalar('Recall/val', recall, epoch)
    writer.add_scalar('F1 Score/val', f1, epoch)

    if epoch > 15 and accuracy > best_accuracy:
        best_accuracy = accuracy
        model_save_path = f'best_model_epoch_{epoch + 1}.pth'
        save_model(model, optimizer, epoch + 1, accuracy, model_save_path)
        logging.info(f'Found better model with accuracy: {accuracy:.4f}. Model saved to {model_save_path}')
        print(f'Found better model with accuracy: {accuracy:.4f}. Model saved to {model_save_path}')

writer.close()

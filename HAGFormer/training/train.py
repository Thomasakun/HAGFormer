import torch
import torch.optim as optim
import logging
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.processing import preprocess_data
from models.model import TransformerClassifier, LabelSmoothingCrossEntropy
from utils.utils import save_model, load_model

def train_model(params, device, train_loader, val_loader, writer):
    """
    Train the Transformer model.

    Args:
        params (dict): Hyperparameters and configuration.
        device (torch.device): Device to run the training on.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
    """
    model = TransformerClassifier(input_dim=params['input_dim'], num_classes=params['num_classes'],
                                  dim_feedforward=params['dim_feedforward'], nhead=params['nhead'],
                                  num_layers=params['num_layers'], dropout=params['dropout']).to(device).to(torch.float64)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    start_epoch = 0
    best_accuracy = 0.0

    # If resuming from a checkpoint
    if params['resume']:
        try:
            start_epoch, best_accuracy = load_model(params['resume'], model, optimizer)
            logging.info(f'Resumed from checkpoint {params["resume"]}, starting at epoch {start_epoch} with best accuracy {best_accuracy:.4f}')
        except Exception as e:
            logging.error(f'Failed to load checkpoint {params["resume"]}: {e}')
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

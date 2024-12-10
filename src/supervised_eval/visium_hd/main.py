import os
import logging
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Dataset Definition
# ---------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, metadata_path, embeddings_dir, split):
        """
        Dataset to load embeddings and labels directly from a folder.

        Args:
        - metadata_path: Path to the metadata CSV file.
        - embeddings_dir: Path to the directory containing extracted .pt files.
        - split: Data split ('train', 'val', or 'test').
        """
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split]
        self.embeddings_dir = embeddings_dir
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(self.metadata['label'].unique()))}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        embedding_path = os.path.join(self.embeddings_dir, str(sample['label']), f"{sample['split']}_{sample['input']}.pt")
        embedding = torch.load(embedding_path)  # Load .pt file directly
        label = self.label_mapping[sample['label']]
        return embedding, label


# ---------------------------
# Step 2: DataModule Definition
# ---------------------------
class EmbeddingDataModule(LightningDataModule):
    def __init__(self, metadata_path, embeddings_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.metadata_path = metadata_path
        self.embeddings_dir = embeddings_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = EmbeddingDataset(self.metadata_path, self.embeddings_dir, 'train')
        self.val_dataset = EmbeddingDataset(self.metadata_path, self.embeddings_dir, 'val')
        self.test_dataset = EmbeddingDataset(self.metadata_path, self.embeddings_dir, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# ---------------------------
# Step 3: Linear Probe Model
# ---------------------------
class LinearProbe(LightningModule):
    def __init__(self, input_dim, num_classes, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
        auroc = roc_auc_score(torch.nn.functional.one_hot(y, self.hparams.num_classes).cpu(), logits.cpu(), average='macro')
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)
        self.log('test_auroc', auroc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


# ---------------------------
# Step 4: Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, required=True, help="Path to the metadata CSV")
    parser.add_argument('--embeddings_dir', type=str, required=True, help="Path to the folder with .pt files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save training logs and models")
    parser.add_argument('--input_dim', type=int, required=True, help="Dimension of the input embeddings")
    parser.add_argument('--num_classes', type=int, required=True, help="Number of output classes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--max_epochs', type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    # DataModule
    data_module = EmbeddingDataModule(args.metadata_csv, args.embeddings_dir, args.batch_size)

    # Model
    model = LinearProbe(args.input_dim, args.num_classes)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger(save_dir=args.output_dir),
        callbacks=[LearningRateMonitor(logging_interval='step'), TQDMProgressBar(refresh_rate=10)],
        accelerator="auto",
    )

    # Train and test
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()

# import os
# import logging
# import argparse
# import torch
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# import pytorch_lightning as pl
# from pytorch_lightning import LightningDataModule, LightningModule
# from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
# from pytorch_lightning.loggers import CSVLogger
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
# import matplotlib.pyplot as plt


# # ---------------------------
# # Step 1: Dataset Definition
# # ---------------------------
# class EmbeddingDataset(Dataset):
#     def __init__(self, metadata_path, embeddings_dir, split):
#         self.metadata = pd.read_csv(metadata_path)
#         self.metadata = self.metadata[self.metadata['split'] == split]
#         self.embeddings_dir = embeddings_dir
#         self.label_mapping = {label: idx for idx, label in enumerate(sorted(self.metadata['label'].unique()))}

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         sample = self.metadata.iloc[idx]
#         embedding_path = os.path.join(self.embeddings_dir, str(sample['label']), f"{sample['split']}_{sample['input']}.pt")
#         embedding = torch.load(embedding_path)
#         label = self.label_mapping[sample['label']]
#         return embedding, label


# # ---------------------------
# # Step 2: DataModule Definition
# # ---------------------------
# class EmbeddingDataModule(LightningDataModule):
#     def __init__(self, metadata_path, embeddings_dir, batch_size=64, num_workers=4):
#         super().__init__()
#         self.metadata_path = metadata_path
#         self.embeddings_dir = embeddings_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def setup(self, stage=None):
#         self.train_dataset = EmbeddingDataset(self.metadata_path, self.embeddings_dir, 'train')
#         self.val_dataset = EmbeddingDataset(self.metadata_path, self.embeddings_dir, 'val')
#         self.test_dataset = EmbeddingDataset(self.metadata_path, self.embeddings_dir, 'test')

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# # ---------------------------
# # Step 3: Linear Probe Model
# # ---------------------------
# class LinearProbe(LightningModule):
#     def __init__(self, input_dim, num_classes, lr=1e-3, weight_decay=1e-4):
#         super().__init__()
#         self.save_hyperparameters()
#         self.linear = torch.nn.Linear(input_dim, num_classes)
#         self.criterion = torch.nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.linear(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy_score(y.cpu(), preds.cpu())
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy_score(y.cpu(), preds.cpu())
#         f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
#         self.log('val_loss', loss, prog_bar=True)
#         self.log('val_acc', acc, prog_bar=True)
#         self.log('val_f1', f1, prog_bar=True)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         preds = torch.argmax(logits, dim=1)
#         acc = accuracy_score(y.cpu(), preds.cpu())
#         f1 = f1_score(y.cpu(), preds.cpu(), average='weighted')
#         auroc = roc_auc_score(torch.nn.functional.one_hot(y, self.hparams.num_classes).cpu(), logits.cpu(), average='macro')
#         self.log('test_acc', acc, prog_bar=True)
#         self.log('test_f1', f1, prog_bar=True)
#         self.log('test_auroc', auroc, prog_bar=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         return optimizer


# # ---------------------------
# # Step 4: Visualization Callback
# # ---------------------------
# class MetricVisualizationCallback(pl.Callback):
#     def __init__(self, output_dir):
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)

#     def on_train_end(self, trainer, pl_module):
#         print(f"MetricVisualizationCallback triggered. Output dir: {self.output_dir}")
#         metrics_path = os.path.join(self.output_dir, 'metrics.csv')
#         if not os.path.exists(metrics_path):
#             print(f"Metrics file not found at {metrics_path}")
#             return

#         metrics = pd.read_csv(metrics_path)
#         print("Metrics file loaded successfully. Generating plots...")
#         plt.figure(figsize=(10, 5))

#         # Plot Loss
#         plt.plot(metrics['epoch'], metrics['train_loss_epoch'], label='Train Loss')
#         plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.title('Loss Curve')
#         plt.savefig(os.path.join(self.output_dir, 'loss_curve.png'))
#         print(f"Loss curve saved to {self.output_dir}/loss_curve.png")
#         plt.close()

#         # Plot Accuracy
#         plt.figure(figsize=(10, 5))
#         plt.plot(metrics['epoch'], metrics['train_acc_epoch'], label='Train Accuracy')
#         plt.plot(metrics['epoch'], metrics['val_acc'], label='Validation Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()
#         plt.title('Accuracy Curve')
#         plt.savefig(os.path.join(self.output_dir, 'accuracy_curve.png'))
#         print(f"Accuracy curve saved to {self.output_dir}/accuracy_curve.png")
#         plt.close()


# # ---------------------------
# # Step 5: Main Function
# # ---------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--metadata_csv', type=str, required=True, help="Path to the metadata CSV")
#     parser.add_argument('--embeddings_dir', type=str, required=True, help="Path to the folder with .pt files")
#     parser.add_argument('--output_dir', type=str, required=True, help="Directory to save training logs and models")
#     parser.add_argument('--input_dim', type=int, required=True, help="Dimension of the input embeddings")
#     parser.add_argument('--num_classes', type=int, required=True, help="Number of output classes")
#     parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
#     parser.add_argument('--max_epochs', type=int, default=10, help="Number of training epochs")
#     args = parser.parse_args()

#     # DataModule
#     data_module = EmbeddingDataModule(args.metadata_csv, args.embeddings_dir, args.batch_size)

#     # Model
#     model = LinearProbe(args.input_dim, args.num_classes)

#     # Trainer
#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         logger=CSVLogger(save_dir=args.output_dir),
#         callbacks=[
#             LearningRateMonitor(logging_interval='step'),
#             TQDMProgressBar(refresh_rate=10),
#             MetricVisualizationCallback(output_dir=args.output_dir),  # Add this callback
#         ],
#         accelerator="auto",
#     )

#     # Train and test
#     trainer.fit(model, datamodule=data_module)
#     trainer.test(model, datamodule=data_module)


# if __name__ == "__main__":
#     main()
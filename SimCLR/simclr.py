import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
import torchmetrics
import torch.nn as nn
import torchvision
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
# TODO try to use pl_bolts.models.self_supervised.simclr.simclr_module

from resnet_simclr import ResNetSimCLR
from simclr_dataset import SimCLRDataset

class SimCLR(pl.LightningModule):
    def __init__(self, batch_size, num_samples, tau=.05):
        super().__init__()
        
        self.simclr_model = ResNetSimCLR(out_dim=128) # paper found a latent space of 128 is best, but doesnt matter much

        self.loss_fn = nn.CrossEntropyLoss()

        self.tau = tau # temperature scaling parameter
        self.batch_size = batch_size
        self.train_iters_per_epoch = num_samples // batch_size

    def generate_nce_targets(self, features):
        # Normalized temperature-scaled CE loss
        # Cross entropy loss of a similarity matrix (divided by a constant scalar temperature)
        # with GT simialirty being 1 if the pairing is positive
        # ie an augmented image pair x1,x2 generated from the same image x will have GT similarity score 1  
        # Similarity function used is cosine similarity

        # Gives a matrix that is the identiy matrix of size N (N=batch size) that is repeated 4 times to
        # give a tiling of identity matrices that totals size (2N x 2N):
        # [I, I]
        # [I, I]
        # This corresponds to the positive lables in the paper (same base image, different augmentations)
        labels = torch.cat([torch.arange(features.shape[0] // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)
        
        # Cosine similarity
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Denote a positive pair as x1,x2
        # Remove main diagonal from both matrices (note that we still want both pairing, x1->x2 and x2->x1)
        # Before, the main diagonal corresponded to x1->x1 and x2->x2
        # The top right identity matrix corresponded to x1->x2 and bottom left corresponded to x2->x1
        # After removing the main diagonal, we only have the positive pairings, x1,x2 and x2,x1
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1) # shape is now 2N x 2N-1 since we removed the main diagonal (N elements)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Temperature scaling
        logits /= self.tau

        return logits, labels

    # write the training step for a simclr model
    def training_step(self, batch, batch_idx):
        images = torch.cat(batch, dim=0)

        features = self.simclr_model(images)
        logits, labels = self.generate_nce_targets(features)
        loss = self.loss_fn(logits, labels)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = torch.cat(batch, dim=0)

        features = self.simclr_model(images)
        logits, labels = self.generate_nce_targets(features)
        loss = self.loss_fn(logits, labels)

        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = .075 * self.batch_size ** .5
        optimizer = LARS(self.parameters(), lr=lr)
        warmup_steps = self.train_iters_per_epoch * 10 # 10 warmup epochs
        total_steps = self.train_iters_per_epoch * self.trainer.max_epochs # 100 total epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

if __name__ == '__main__':
    import torchvision.transforms as transforms
    import numpy as np
    import os
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    size = 256

    # Transformations according to paper
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    blur = transforms.GaussianBlur(kernel_size=int(0.1 * size), sigma=np.random.uniform(0.1, 2.0))
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((size, size)),
                                            transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(p=.5),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.RandomApply([blur], p=0.5),
                                            ])
    dataset_dirs = []
    for filename in os.listdir('../images'):
        if filename.endswith('.jpg'):
            img_path = os.path.join('../images', filename)
            dataset_dirs.append(img_path)
    for filename in os.listdir('../images/Classification/JPEGImages/'):
        if filename.endswith('.jpg'):
            img_path = os.path.join('../images/Classification/JPEGImages/', filename)
            dataset_dirs.append(img_path)

    dataset = SimCLRDataset(dataset_dirs, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    tb_logger = pl_loggers.TensorBoardLogger("log/")
    checkpoint_callback = pl_callbacks.ModelCheckpoint("ckpt/", monitor="val/loss")

    model = SimCLR(batch_size=16, num_samples=len(dataset_dirs), tau=.05)
    trainer = pl.Trainer(gpus=1, max_epochs=100, logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, loader)
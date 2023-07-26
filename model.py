from torch import nn, optim
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Metric
import lightning.pytorch as pl
import torchvision

# Create the Lightning module
class LightningModel(pl.LightningModule):
    # Model definition as in PyTorch (arch. + forward)
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, num_classes)

        self.loss_f = nn.CrossEntropyLoss()

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x_ = F.relu(self.l2(x))
        x = F.relu(self.l3(x_))
        x = F.relu(self.l4(x + x_))
        x = self.l5(x)
        return x

    # Training procedure
    def training_step(self, batch, batch_idx):
        x, y_true = batch
        x = x.view(x.size(0), -1) 
        y_pred = self(x)
        loss = self.loss_f(y_pred, y_true)
        self.log('Train Loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        # Example images to Tensorboard
        if batch_idx % 100 == 0:
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image('Example EMNIST training images', grid, self.global_step)

        return loss

    # Validation procedure
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        x = x.view(x.size(0), -1)
        y_pred = self(x)
        loss = self.loss_f(y_pred, y_true)
        acc = self.acc(y_pred, y_true)
        self.log_dict({'Val. Loss': loss, 'Val. Acc.': acc})
        return loss
        
    # Testing procedure
    def test_step(self, batch, batch_idx):
        x, y_true = batch
        x = x.view(x.size(0), -1)
        y_pred = self(x)
        loss = self.loss_f(y_pred, y_true)
        acc = self.acc(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        self.log_dict({'Test Recall': recall, 'Test Acc.': acc})
        return loss
    
    # Optimizer (and LR if needed)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
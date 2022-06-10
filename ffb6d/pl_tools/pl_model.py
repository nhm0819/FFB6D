# Weights & Biases
import wandb

# Pytorch modules
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

# Pytorch-Lightning
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torchmetrics
from ffb6d.common import Config, ConfigRandLA
from ffb6d.models.ffb6d import FFB6D
from ffb6d.models.loss import OFLoss, FocalLoss


class pl_ffb6d(LightningModule):

    def __init__(self, args, lr=1e-3, pretrained=False):
        '''method used to define our model parameters'''
        super().__init__()

        self.args = args
        config = Config(ds_name='neuromeka', cls_type=args.cls)
        rndla_cfg = ConfigRandLA
        model = FFB6D(
            n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
            n_kps=config.n_keypoints
        )

        self.minibatch_per_epoch = 70000 // self.args.train_bs

        if self.args.gpus > 1 or self.args.gpus == -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # optimizer parameters
        self.lr = self.args.lr

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        # x = self.softmax(x)
        return x


    # convenient method to get the loss on a batch
    def loss(self, x, y):
        logits = self(x)  # this calls self.forward
        loss = -F.nll_loss(logits, y)
        return logits, loss


    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        x, y = batch
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        logits, loss = self.loss(x, y)

        preds = torch.argmax(logits, 1)

        # Log training loss
        self.log('train_loss', loss)

        # Log metrics
        self.log('train_acc', self.accuracy(preds, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.log("val_loss", loss)  # default on val/test is on_epoch only
        self.log('val_acc', self.accuracy(preds, y))

        return logits

    def validation_epoch_end(self, validation_step_outputs):
        dummy_input = torch.zeros((3, 320, 320), device=self.device)
        model_filename = f"model_{str(self.global_step).zfill(5)}.pt"
        self.to_torchscript(model_filename, method="script", example_inputs=dummy_input)
        wandb.save(model_filename)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid_logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.accuracy(preds, y), on_step=False, on_epoch=True)

    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        dummy_input = torch.zeros((3, 320, 320), device=self.device)
        model_filename = "model_final.pt"
        # self.to_onnx(model_filename, dummy_input, export_params=True)
        self.to_torchscript(model_filename, method="script", example_inputs=dummy_input)
        wandb.save(model_filename)

        flattened_logits = torch.flatten(torch.cat(test_step_outputs))
        self.logger.experiment.log(
            {"test_logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})


    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = CyclicLR(
            optimizer, base_lr=1e-5, max_lr=1e-3,
            cycle_momentum=False,
            step_size_up=self.args.epochs * self.minibatch_per_epoch // 2 // self.args.gpus,
            step_size_down=self.args.epochs * self.minibatch_per_epoch // 2 // self.args.gpus,
            mode='triangular'
        )

        return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "train_loss"},
        }
# Weights & Biases
import wandb

# basic
import numpy as np

# Pytorch modules
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

# Pytorch-Lightning
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torchmetrics

# ffb6d
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from models.loss import OFLoss, FocalLoss
from utils.pvn3d_eval_utils_kpls import TorchEval


class pl_ffb6d(LightningModule):

    def __init__(self, config=None, ds_name="neuromeka", cls_type="bottle", data_length=80150,
                 batch_size=4, lr=1e-3, weight_decay=0, epochs=10, gpus=-1):
        '''method used to define our model parameters'''
        super().__init__()

        self.config = config
        self.ds_name = ds_name
        self.cls_type = cls_type
        self.batch_size = batch_size
        self.lr = lr
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.epochs = epochs

        config = Config(ds_name=self.ds_name, cls_type=cls_type)
        rndla_cfg = ConfigRandLA
        self.model = FFB6D(
            n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
            n_kps=config.n_keypoints
        )

        self.minibatch_per_epoch = data_length // self.batch_size
        self.cls_div = 2

        if gpus > 1 or torch.cuda.device_count() > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # optimizer parameters
        self.lr = lr

        # loss
        self.focal_loss = FocalLoss(gamma=2)
        self.of_loss = OFLoss()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

        self.torch_eval = TorchEval()

    def forward(self, input):
        """
        model Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            choose      : LongTensor [bs, 1, npts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns: dict of :
            pred_rgbd_segs  : FloatTensor --> Focal Loss(pred_rgbd_segs, labels.view(-1))
            pred_kp_ofs     : FloatTensor --> OF Loss(pred_kp_ofs, kp_targ_ofst, labels)
            pred_ctr_ofs    : FloatTensor --> OF Loss(pred_ctr_ofs, ctr_targ_ofst, labels)
        """
        end_points = self.model(input)
        return end_points


    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        data = batch
        cu_dt = {}
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()

        end_points = self(cu_dt)

        labels = cu_dt['labels']
        loss_rgbd_seg = self.focal_loss(
            end_points['pred_rgbd_segs'], labels.view(-1)
        ).sum()
        loss_kp_of = self.of_loss(
            end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst'], labels
        ).sum()
        loss_ctr_of = self.of_loss(
            end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst'], labels
        ).sum()

        loss_lst = [
            (loss_rgbd_seg, 2.0), (loss_kp_of, 1.0), (loss_ctr_of, 1.0),
        ]
        loss = sum([ls * w for ls, w in loss_lst])

        # Log training loss
        self.log('train_loss_rgbd_seg', loss_rgbd_seg.item())
        self.log('train_loss_kp_of', loss_kp_of.item())
        self.log('train_loss_ctr_of', loss_ctr_of.item())
        self.log('train_loss', loss.item())

        # Log metrics
        _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
        acc_rgbd = (cls_rgbd == labels).float().sum() / labels.numel()
        self.log('train_acc', acc_rgbd.item())

        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        cu_dt = {}
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()

        end_points = self(cu_dt)

        labels = cu_dt['labels']
        loss_rgbd_seg = self.focal_loss(
            end_points['pred_rgbd_segs'], labels.view(-1)
        ).sum()
        loss_kp_of = self.of_loss(
            end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst'], labels
        ).sum()
        loss_ctr_of = self.of_loss(
            end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst'], labels
        ).sum()

        loss_lst = [
            (loss_rgbd_seg, 2.0), (loss_kp_of, 1.0), (loss_ctr_of, 1.0),
        ]
        loss = sum([ls * w for ls, w in loss_lst])

        # Log training loss
        self.log('val_loss_rgbd_seg', loss_rgbd_seg.item())
        self.log('val_loss_kp_of', loss_kp_of.item())
        self.log('val_loss_ctr_of', loss_ctr_of.item())
        self.log('val_loss', loss.item())

        # Log metrics
        _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
        acc_rgbd = (cls_rgbd == labels).float().sum() / labels.numel()
        self.log('val_acc', acc_rgbd.item())

        return end_points

    # def validation_epoch_end(self, validation_step_outputs):
    #     save_path = f"train_log/{self.ds_name}/ckpt/model_{str(self.global_step).zfill(5)}.pt"
    #     self.to_torchscript(save_path, method="script")
    #
    #     pred_rgbd_segs_logits = torch.flatten(torch.cat(validation_step_outputs['pred_rgbd_segs']))
    #     pred_kp_ofs_logits = torch.flatten(torch.cat(validation_step_outputs['pred_kp_ofs']))
    #     pred_ctr_ofs_logits = torch.flatten(torch.cat(validation_step_outputs['pred_ctr_ofs']))
    #     self.logger.experiment.log(
    #         {"pred_rgbd_segs_logits": wandb.Histogram(pred_rgbd_segs_logits.to("cpu")),
    #          "pred_kp_ofs_logits": wandb.Histogram(pred_kp_ofs_logits.to("cpu")),
    #          "pred_ctr_ofs_logits": wandb.Histogram(pred_ctr_ofs_logits.to("cpu")),
    #          "global_step": self.global_step})


    def test_step(self, batch, batch_idx):
        data = batch
        cu_dt = {}
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()

        end_points = self(cu_dt)

        labels = cu_dt['labels']
        loss_rgbd_seg = self.focal_loss(
            end_points['pred_rgbd_segs'], labels.view(-1)
        ).sum()
        loss_kp_of = self.of_loss(
            end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst'], labels
        ).sum()
        loss_ctr_of = self.of_loss(
            end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst'], labels
        ).sum()

        loss_lst = [
            (loss_rgbd_seg, 2.0), (loss_kp_of, 1.0), (loss_ctr_of, 1.0),
        ]
        loss = sum([ls * w for ls, w in loss_lst])

        # Log training loss
        self.log('test_loss_rgbd_seg', loss_rgbd_seg.item())
        self.log('test_loss_kp_of', loss_kp_of.item())
        self.log('test_loss_ctr_of', loss_ctr_of.item())
        self.log('test_loss', loss.item())

        # Log metrics
        _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
        acc_rgbd = (cls_rgbd == labels).float().sum() / labels.numel()
        self.log('test_acc', acc_rgbd.item())


    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = CyclicLR(
            optimizer, base_lr=1e-5, max_lr=1e-3,
            cycle_momentum=False,
            step_size_up=self.epochs * self.minibatch_per_epoch // self.cls_div // self.gpus,
            step_size_down=self.epochs * self.minibatch_per_epoch // self.cls_div // self.gpus,
            mode='triangular'
        )

        # TODO : To Apply Batch Normalization Scheduler

        return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "train_loss"},
        }
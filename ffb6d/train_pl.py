# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

# Pytorch modules
import torch

# Pytorch-Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.plugins import DDPPlugin

from pl_tools.pl_model import pl_ffb6d
from pl_tools.pl_datamodule import NeuromekaDataModule

# ffb6d
from common import Config, ConfigRandLA
from utils.basic_utils import Basic_Utils

# basic
import argparse
import random
import os
import numpy as np



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Classification')
parser.add_argument('--dataset', type=str, default="neuromeka", metavar='S',
                    help='dataset name')
parser.add_argument('--cls', type=str, default="bottle", metavar='S',
                    help='class name')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='train batch size / num_gpus')
parser.add_argument('--checkpoint', type=int, default=5, metavar='N',
                    help='checkpoint period')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers')
parser.add_argument('--gpus', type=int, default=1, metavar='N',
                    help='number of gpus')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                    help='optimizer weight decay')




def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # seed_torch(42)
    args = parser.parse_args()
    cls_type = args.cls
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')


    config = Config(ds_name=args.dataset, cls_type=args.cls)

    wandb_project = f"neuromeka-{cls_type}"
    wandb.init()
    # config = wandb.config

    wandb_logger = WandbLogger(
        project=wandb_project, name=now, log_model=False, save_dir=f"train_log/{now}/wandb"
    )
    tb_logger = TensorBoardLogger(save_dir=f"train_log/{now}/tensorboard")

    # callback lists
    MODEL_CKPT_PATH = f'train_log/{args.dataset}/ckpt/{cls_type}/{now}'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    ealry_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', dirpath=MODEL_CKPT_PATH, filename=MODEL_CKPT,
        save_top_k=args.checkpoint, mode='min')

    # setup data
    neuromeka_data = NeuromekaDataModule(
        config=config,
        ds_name=args.dataset,
        cls_type=args.cls,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    neuromeka_data.prepare_data()
    neuromeka_data.setup()

    # setup model
    model = pl_ffb6d(config=config, ds_name=args.dataset, cls_type=args.cls,
                     data_length=len(neuromeka_data.train_ds), batch_size=args.batch_size, lr=args.lr,
                     weight_decay=args.weight_decay, epochs=args.epochs, gpus=args.gpus)

    wandb_logger.watch(model, log_freq=500)

    # grab samples to log predictions on
    samples = next(iter(neuromeka_data.test_dataloader()))

    trainer = pl.Trainer(
        logger=[wandb_logger,tb_logger],    # W&B integration
        log_every_n_steps=500,   # set the logging frequency
        val_check_interval=0.2,
        gpus=args.gpus,                # use all GPUs
        max_epochs=args.epochs // args.gpus, # number of epochs
        # deterministic=True,     # keep it deterministic
        callbacks=[ealry_stop_callback,
                   checkpoint_callback], # see Callbacks section
        precision=16,
        check_val_every_n_epoch=args.checkpoint,
        strategy= 'ddp', # "ddp_find_unused_parameters_false",
        # plugins=DDPPlugin,
        auto_scale_batch_size=True,
        profiler='pytorch' # 'simple'
    )

    # fit the model
    trainer.fit(model, neuromeka_data)

    # # evaluate the model on a test set
    # trainer.test(datamodule=neuromeka_data)

    wandb.finish()



if __name__ == "__main__":
    main()


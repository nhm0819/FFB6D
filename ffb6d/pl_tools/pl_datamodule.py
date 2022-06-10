from pytorch_lightning import LightningDataModule
import os
from ffb6d.datasets.neuromeka.neuromeka_dataset import Dataset as Neuromeka_Dataset
from ffb6d.common import Config, ConfigRandLA
from ffb6d.utils.basic_utils import Basic_Utils
from torch.utils.data import DataLoader


class NeuromekaDataModule(LightningDataModule):
    def __init__(self, args=None, batch_size=16, num_workers=0):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers

        if args.dataset == "ycb":
            self.config = Config(ds_name=args.dataset)
        elif args.dataset == "neuromeka":
            self.config = Config(ds_name=args.dataset, cls_type=args.cls)
        else:
            self.config = Config(ds_name=args.dataset, cls_type=args.cls)
        self.bs_utils = Basic_Utils(self.config)


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.train_ds = Neuromeka_Dataset('train', cls_type=self.args.cls)
            self.val_ds = Neuromeka_Dataset('test', cls_type=self.args.cls)
        if stage == 'test' or stage is None:
            self.test_ds = Neuromeka_Dataset('test', cls_type=self.args.cls)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.config.mini_batch_size,
                                  shuffle=False, drop_last=True, num_workers=0, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=self.config.val_mini_batch_size,
                                shuffle=False, drop_last=False, num_workers=0)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=2 * self.batch_size, num_workers=self.num_workers)
        return test_loader


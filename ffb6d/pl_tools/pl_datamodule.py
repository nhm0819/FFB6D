from pytorch_lightning import LightningDataModule
import os
from datasets.neuromeka.neuromeka_dataset import Dataset as Neuromeka_Dataset
from common import Config, ConfigRandLA
from utils.basic_utils import Basic_Utils
from torch.utils.data import DataLoader


class NeuromekaDataModule(LightningDataModule):
    def __init__(self, config=None, bs_utils=None, ds_name="neuromeka", cls_type="bottle", batch_size=4, num_workers=0):
        super().__init__()
        self.ds_name = ds_name
        self.cls = cls_type
        self.batch_size = batch_size
        self.num_workers = num_workers

        if config is not None:
            self.config = config
        else:
            if ds_name == "ycb":
                self.config = Config(ds_name=ds_name)
            elif ds_name == "neuromeka":
                self.config = Config(ds_name=ds_name, cls_type=cls_type)
            else:
                self.config = Config(ds_name=ds_name, cls_type=cls_type)
        if bs_utils is None:
            self.bs_utils = Basic_Utils(self.config)
        else:
            self.bs_utils = bs_utils


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.train_ds = Neuromeka_Dataset('train', cls_type=self.cls, batch_size=self.batch_size)
            self.val_ds = Neuromeka_Dataset('test', cls_type=self.cls, batch_size=self.batch_size)
        if stage == 'test' or stage is None:
            self.test_ds = Neuromeka_Dataset('test', cls_type=self.cls, batch_size=2*self.batch_size)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size,
                                  shuffle=False, drop_last=True,
                                  num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size,
                                shuffle=False, drop_last=False,
                                num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds, batch_size=2*self.batch_size,
                                 num_workers=self.num_workers)
        return test_loader


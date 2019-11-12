import os, sys
import importlib
import warnings
import numpy as np
import pandas as pd
import torch
from addict import Dict

class AracleConfig:
    """Nested dictionary representing the configuration for H0rton training, h0_inference, visualization, and analysis

    """
    def __init__(self, user_cfg):
        """
        Parameters
        ----------
        user_cfg : dict or Dict
            user-defined configuration
        
        """
        self.__dict__ = Dict(user_cfg)
        self.validate_user_definition()
        self.preset_default()
        self.set_device()
        # Data
        self.set_XY_metadata()        
        self.set_model_metadata()

    @classmethod
    def from_file(cls, user_cfg_path):
        """Alternative constructor that accepts the path to the user-defined configuration python file

        Parameters
        ----------
        user_cfg_path : str or os.path object
            path to the user-defined configuration python file

        """
        dirname, filename = os.path.split(os.path.abspath(user_cfg_path))
        module_name, ext = os.path.splitext(filename)
        sys.path.append(dirname)
        #user_cfg_file = map(__import__, module_name)
        #user_cfg = getattr(user_cfg_file, 'cfg')
        user_cfg_script = importlib.import_module(module_name)
        user_cfg = getattr(user_cfg_script, 'cfg')
        return cls(user_cfg)

    def validate_user_definition(self):
        """Check to see if the user-defined config is valid

        """
        pass
        
    def preset_default(self):
        """Preset default config values

        """
        pass

    def set_device(self):
        """Configure the device to use for training

        Note
        ----
        Disabling fallback to cpu when cuda is unavailable, for full reproducibility.

        """
        # Disable this check for reproducibility
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_type)
        if self.device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    def set_XY_metadata(self):
        """Set general metadata relevant to network architecture and optimization

        """
        pass

    def set_model_metadata(self):
        """Set metadata about the network architecture and the loss function (posterior type)

        """
        if self.model.load_pretrained:
            # Pretrained model expects exactly this normalization
            self.data.X_mean = [0.485, 0.456, 0.406]
            self.data.X_std = [0.229, 0.224, 0.225]

    def check_train_val_diff(self):
        """Check that the training and validation datasets are different

        """
        pass
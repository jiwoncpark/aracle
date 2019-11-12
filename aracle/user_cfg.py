import numpy as np
from addict import Dict

cfg = Dict()

# Global configs
cfg.device_type = 'cpu'
cfg.global_seed = 1225

# Data
cfg.data = Dict(train_dir='/home/jwp/stage/aracle/my_data',
                val_dir='/home/jwp/stage/aracle/my_data',
                t_offset=5,
                normalize_X=True,
                X_mean=[0.485, 0.456, 0.406],
                X_std=[0.229, 0.224, 0.225],
                X_dim=224,
                plot_idx=np.arange(100),
                )

# Model
cfg.model = Dict(
                 load_pretrained=True,
                 )

# Optimization
cfg.optim = Dict(n_epochs=2,
                 learning_rate=1.e-4,
                 batch_size=5,
                 lr_scheduler=Dict(milestones=[50, 90],
                                   gamma=0.7))

# Logging
cfg.log = Dict(checkpoint_dir='saved_models', # where to store saved models
               checkpoint_interval=1, # in epochs
               logging_interval=1, # in epochs
               monitor_1d_marginal_mapping=False,
               )
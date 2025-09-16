from types import SimpleNamespace
from pathlib import Path

DATAPATH_PREFIX = Path('data') # prefix to shorten the paths
DATAPATHS=[
    'data_369.pkl',
    'data_370.pkl',
    'data_371.pkl',
    'data_372.pkl',
    'data_373.pkl',
    'data_375.pkl',
    'data_376.pkl',
    'data_377.pkl',
    'data_378.pkl',
    'data_379.pkl',
    'data_380.pkl',
    'data_381.pkl',
    'data_382.pkl',
    'data_383.pkl',
    'data_452.pkl',
    'data_452.pkl',
    'data_453.pkl',
    'data_454.pkl',
    'data_455.pkl',
] # Paths to the input data

cfg = {
    'data': {
        'dataset': 'kungsbacka', # name of dataset
        'preprocessed_datapath':DATAPATH_PREFIX / 'bus_data_processed.pkl', # Path to store training data
        'datapaths':[DATAPATH_PREFIX/'4_years'/path for path in DATAPATHS], # List of paths to pickled pandas or csv
        'traj_length': 200,
        'num_workers': True,
        'channels': 2, # legacy
        'traj_path1': './data/trips.pkl', # legacy
        'head_path2': './data/trip_heads.pkl', # legacy
        'uniform_dequantization': False, # legacy
        'gaussian_dequantization': False,# legacy
    },
    'model': {
        'type': "simple",
        'attr_dim': 8,
        'guidance_scale': 3,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
        'dropout': 0.1,
        'var_type': 'fixedlarge',
        'ema_rate': 0.998, # https://arxiv.org/abs/2411.18704
        'ema': True,
        'resamp_with_conv': True,
        'learning_rate': 2e-4,
    },
    'diffusion': {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.05,
        'num_diffusion_timesteps': 500,
    },
    'training': {
        'batch_size': 512,
        'n_epochs': 2000,
        'n_iters': 5000000, # legacy
        'snapshot_freq': 5000, # legacy
        'validation_freq': 2000, # legacy
        'GPUs': [], # leave empty to disable DDP
        'use_triton': False, # greatly speeds up code if installed
        'checkpoint': '' # leave empty to make new
    },
    'sampling': { # not used
        'batch_size': 64,
        'last_only': True,
    }
}

temp = {}
for k, v in cfg.items():
    temp[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**temp)
from types import SimpleNamespace

cfg = {
    'data': {
        'dataset': 'kungsbacka',
        'traj_path1': './data/trips.pkl',
        'head_path2': './data/trip_heads.pkl',
        'traj_length': 200,
        'channels': 2,
        'uniform_dequantization': False,
        'gaussian_dequantization': False,
        'num_workers': True,
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
        'batch_size': 256,
        'n_epochs': 2000,
        'n_iters': 5000000,
        'snapshot_freq': 5000,
        'validation_freq': 2000,
        'GPUs': [],
        'use_triton': True,
    },
    'sampling': {
        'batch_size': 64,
        'last_only': True,
    }
}

temp = {}
for k, v in cfg.items():
    temp[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**temp)
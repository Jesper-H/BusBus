"""
Main functionality of the code base goes here along with command line bindings
"""
import sys
import importlib
import importlib.util
import fire
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from pathlib import Path
from tqdm import tqdm
from itertools import islice

# Local imports
sys.path.append('.')
from busbus.difftraj.helper import make_beta_schedule
from busbus.difftraj.Traj_UNet import Guide_UNet
from busbus.difftraj.utils import p_xt
from busbus.dataloader import save_trajectories, dataloader_from_cfg
from busbus.preprocessing import trip_normalizer
from busbus.utils import no_return


def create_model(cfg):
    model = Guide_UNet(cfg).cuda()

    # Create/Load model data
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.learning_rate)
    epoch = 0
    losses = []
    if cfg.training.checkpoint:
        checkpoint = torch.load(cfg.training.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']

    # Add model wrappers
    do_triton = importlib.util.find_spec("triton") is not None
    do_triton &= cfg.training.use_triton == True
    if do_triton:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    if cfg.training.use_triton and not do_triton:
        print('Warning: no triton installation found, skipping. Disable triton in config to hide this')
        
    if cfg.training.GPUs:
        model = DDP(model, device_ids=[device_id]) # DDP does not work in jupyter notebook :(
        
    return model, optimizer, epoch, losses

def inference(model_path:str, num_batches:int, save_path:str=None, intermediates:bool=False) -> list[pd.DataFrame]:
    "Generates new trajectories"
    # Get relative path from model to its config file
    model_path = Path(model_path)
    config_path = [*model_path.parts[:-1]] 
    config_path = config_path[:-2] + ['Files', config_path[-1], 'config.py'] 
    config_path = Path(*config_path)
    
    # Load the config module in a windows compatible way
    spec = importlib.util.spec_from_file_location('config.py', config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['config'] = module
    spec.loader.exec_module(module)
    cfg = module.cfg
    
    # Declare variables
    n_steps = cfg.diffusion.num_diffusion_timesteps
    beta = make_beta_schedule(
        cfg.diffusion.beta_schedule,
        cfg.diffusion.num_diffusion_timesteps,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end).cuda()
    beta = torch.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    eta = 0.0
    timesteps = 100
    skip = n_steps // timesteps
    seq = range(0, n_steps, skip)
    
    # Load model
    cfg.training.checkpoint = model_path
    cfg.training.GPUs = False # TODO: add ddp support for inference
    unet, *_ = create_model(cfg)
    unet = unet.cuda()

    # Load data
    dataloader, normalizer = dataloader_from_cfg(cfg)

    # Generate
    gen_traj = []
    for _, head in tqdm(islice(dataloader, num_batches), total=num_batches):
        head = head.cuda()
        head[:,-1]=0 # FuelRate=0/1 (good/bad)
        
        # Inital random noise
        x = torch.randn(cfg.training.batch_size, cfg.model.in_channels, cfg.data.traj_length).cuda()
        ims = []
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(seq[::-1], seq_next[::-1]):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            with torch.no_grad():
                pred_noise = unet(x, t, head)
                x = p_xt(x, pred_noise, t, next_t, beta, eta)
                if i % 10 == 0:
                    ims.append(x.cpu())

        head = head.cpu().numpy()
        if intermediates:
            return [normalizer.numpy2trips(im, head) for im in ims]

        trajs = ims[-1].cpu().numpy()
        trajs = normalizer.numpy2trips(trajs.swapaxes(1,2), head)
        gen_traj.extend(trajs)

    if save_path:
        save_trajectories(gen_traj, save_path)
    return gen_traj

if __name__ == '__main__':
    fire.Fire({
        'inference':no_return(inference),
        'train':lambda x: None, # TODO
    })
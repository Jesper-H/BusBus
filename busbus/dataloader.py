"""
Put all data loading/saving code here
"""

import h5py
import json
import pandas as pd
import numpy as np
import torch
import types
import pickle
from itertools import islice
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from busbus.preprocessing import trip_normalizer

def open_data(path:str) -> 'pd.DataFrame':
    "Maps file extention to loader and calls it"
    ext2loader = {
        '.pickle':lambda p: pd.read_pickle(p),
        '.csv':pd.read_csv,
    }
    _, extention = os.path.splitext(path)
    return ext2loader[extention](path)

def make_dataset(trip_arrays:np.ndarray, trip_heads:np.ndarray):
    #trip_tensor = np.swapaxes(trip_arrays, 1, 2)
    trip_tensor = torch.from_numpy(trip_arrays).float()
    head_tensor = torch.from_numpy(trip_heads).float()
    return TensorDataset(trip_tensor, head_tensor)

def make_dataloader(dataset:torch.utils.data.dataset.TensorDataset, cfg:types.SimpleNamespace):
    kwargs = {'batch_size':cfg.training.batch_size, 'shuffle':True}
    if cfg.training.GPUs:
        kwargs.update({'shuffle':False, 'sampler':DistributedSampler(dataset)})
    return DataLoader(dataset, **kwargs)

def dataloader_from_cfg(cfg:types.SimpleNamespace):
    with open(cfg.data.preprocessed_datapath, 'rb') as file:
            trip_arrays, trip_heads, normalizer = pickle.load(file)

    dataset = make_dataset(
        np.swapaxes(trip_arrays, 1, 2), # swap lon and lat for this dataset
        trip_heads)
    return make_dataloader(dataset, cfg), normalizer

def read_h5(path:str) -> 'pandas.DataFrame':
    with h5py.File(path, "r") as f:
        bus_dfs = []
        for bus_name in f.keys():
            bus_data = f[bus_name]
            columns = {
                "timestamp":bus_data["timestamps"],
                "lat":np.degrees(bus_data["GPS0"]), # TODO verify this is latitude
                "lon":np.degrees(bus_data["GPS1"]),
            }
            df = pd.DataFrame(columns)
            
            bus_dfs.append(df)        
        return pd.concat(bus_dfs, ignore_index=True)

def read_panda_pkl(path:str) -> 'pd.DataFrame':
    df = pd.read_pickle(path)
    name_fix={'lat':'lon','lon':'lat'}
    df.rename(columns=name_fix, inplace=True)
    return df.loc[:,['datetime','lat','lon','FuelRate']]

def open_data(paths:list[str]) -> 'pd.DataFrame':
    "Maps file extention to loader and calls it"
    ext2loader = {
        '.pickle':lambda p: pickle.load(open(p,'rb')),
        '.pkl':read_panda_pkl,
        '.csv':pd.read_csv,
    }
    dfs = []
    for path in paths:
        _, extention = os.path.splitext(path)
        df = ext2loader[extention](path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def save_trajectories(trajectories:list[pd.DataFrame], path:str, overwrite:bool=True) -> None:
    """
    Write trajectories to path as an HDF5 file.

    traj: List of DataFrames (one per trajectory).
    path: Target filename.
    overwrite: If True, an existing file is silently replaced.
    """
    mode = ['x','w'][overwrite]
    with h5py.File(path, mode) as h5:
        h5.attrs["n_trajectories"] = len(trajectories)

        for i, df in enumerate(trajectories):
            grp = h5.create_group(f"traj_{i:06d}")
            rec = df.to_records()
            grp.create_dataset('data', data=rec)
            grp.attrs["column_names"] = json.dumps([str(c) for c in df.columns])
            grp.attrs["index_name"] = json.dumps(df.index.name or None)

def load_trajectories(path:str, max_count:int=0) -> list[pd.DataFrame]:
    """
    Read trajectories previously saved with 'save_trajectories'

    path: Filename to load.
    max_count: Limits the number of data points loaded, 0 means no limit.
    return: list of trajectory DataFrames.
    """
    trajectories = []
    max_count = max_count or None
    with h5py.File(path, "r") as h5:
        for key in islice(sorted(h5.keys()), max_count):
            grp = h5[key]
            arr = grp["data"][()]
            col_names = json.loads(grp.attrs["column_names"])
            idx_name = json.loads(grp.attrs["index_name"]) or 'index'

            # rebuild dataset
            df = pd.DataFrame.from_records(arr)
            df.set_index(idx_name, inplace=True)
            df.columns = col_names
            trajectories.append(df)

    return trajectories
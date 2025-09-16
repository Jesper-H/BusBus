"""
Preprocessing functions and classes for trajectories. 
"""

import pickle
import datetime
import numpy as np
import pandas as pd
import torch
import requests
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import timedelta, datetime
from itertools import repeat
from haversine import haversine_vector

def resample_trajectory(x, length:int=200) -> 'np.ndarray':
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x, features = x.shape
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((features, length))
    for i in range(features):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

def timestamps2datetimes(timestamps):
    ref_timestamp = datetime.fromisoformat("2011-06-16T05:23:00").timestamp()
    datetimes = (np.array(timestamps)/1000+ref_timestamp).astype("datetime64[s]")
    return datetimes

def add_datetime_column(df):
    assert 'timestamps' in df.columns
    ref_timestamp = datetime.fromisoformat("2011-06-16T05:23:00").timestamp()
    datetimes = (df.timestamps/1000+ref_timestamp).astype("datetime64[s]")
    df.loc[:, 'datetime'] = pd.Series(datetimes, index=df.index, name='datetime')
    return df

def add_speed_column(df, sample_rate:float=2.0):
    'Uses haversine to estimate the speed. This operation is inplace'
    speed = df.loc[:,['lat','lon']]
    speed = map(haversine_vector, speed.shift(1).bfill().to_numpy(), speed.to_numpy()) # km
    speed = map(lambda x:x[0], speed)
    speed = pd.Series(speed, index=df.index, name='speed')*1000 / sample_rate # m/s
    df.loc[:,'speed'] = speed
    return df

def linear_interpolation(start:float, end:float, steps:int=1) -> list:
    'Interpolates new values between start and end'
    step_size = (end - start) / (steps + 1)
    return [start + i * step_size for i in range(1, steps + 1)]
    
def interpolate_indices(indices:list[int], max_length:int=200) -> list: 
    'Interpolates new indices where the gap greater than max_length, indices must be sorted in ascending order'
    interpolated_indices = []
    for start_index, end_index in zip(indices, indices[1:]):
        interpolated_indices.append(start_index)
        span = end_index - start_index
        if span > max_length:
            new_values = linear_interpolation(start_index, end_index, steps=span//max_length)
            interpolated_indices += [*map(int, new_values)]

    interpolated_indices.append(indices[-1])
    return interpolated_indices

def bin_embedding(list_of_trips, bins=[16,16]):
    """
    Builds uni-spaced embedding ids in space. 
    Note, outliers mess with the bin sizes. 
    Also accepts bins as input for inference purposes.
    """
    x_bins, y_bins = bins
    d = list_of_trips # shorten name
    x_corr = np.array([d[i].iloc[[0,-1]]['lat'] for i in range(len(d))]).ravel() # start and end x
    y_corr = np.array([d[i].iloc[[0,-1]]['lon'] for i in range(len(d))]).ravel()
    x_binned, x_bins=pd.cut(x_corr, x_bins, labels=False, retbins=True) # bin together
    y_binned, y_bins=pd.cut(y_corr, y_bins, labels=False, retbins=True)
    ids = x_binned*(len(x_bins)-1) + y_binned # hash ids
    ids[ids!=ids] = -1 # replace NaN with magic number
    ids += 1 # offset so NaN is 0
    sid, eid = ids.reshape(-1,2).T # separate start and end points
    return sid, eid, [x_bins, y_bins]
        
def get_elevation_valhalla(df):
    'Does not work. TODO: make it do the thing'
    locations = [{"lat": lat, "lon": lon} for lat, lon in zip(df["lat"], df["lon"])]
    # elevation_interval: 30,
    url = "http://localhost:8002/height"
    response = requests.post(url, json={"range": False, "shape": locations}).json()
    
    elevations = response["height"]
    return elevations
    
def map_match_valhalla(df, sample_rate:float=None):
    'Does map matching. Modifies DataFrame inplace! Specify sample rate if known'
    url = "http://localhost:8002/trace_route"
    locations = [{"lat": lat, "lon": lon} for lat, lon in zip(df["lat"], df["lon"])]
    payload = {
        "shape": locations,
        "costing": "bus",  # ["bus"|"auto"|"bicycle"|"pedestrian"]
        "shape_match": "map_snap",
        "format":"osrm",
        "trace_options":{"search_radius":50}, #"interpolation_distance":30}, 
        "directions_type": "none",
        "alternates": 0,
    }
    
    if sample_rate: # if sample rate is known, leverage it to enchance accuracy
        payload.update({
            "durations": [*repeat(sample_rate, len(locations)-1)],
            "use_timestamps": True,
        })

    try:
        response = requests.post(url, json=payload).json()
    except Exception as e:
        print(e)
        return None

    if response['code'] != 'Ok':
        return None
        
    points = [point['location'] if point else [float('NaN')]*2 for point in response['tracepoints']]
    matched = pd.DataFrame(points, index=df.index, columns=['lon', 'lat'])
    if matched.lat.isna().mean() > 0.1: 
        return None
        
    matched.drop_duplicates(keep=False, inplace=True) # merged points are duplicates
    merged_points_ratio = 1.0-(len(matched)/len(points))
    if merged_points_ratio > 0.5:
        return None
        
    df.update(matched) # only update the points that matched
    df = add_speed_column(df) # update speed values
    return df

def process_trip(df, start, end, sample_rate=2.0):
    """Processes a single trip: extracts, matches, and validates it."""
    if start == end:
        return None  # Skip invalid boundaries
    
    trip = df.iloc[start:end].copy().reset_index()

    if len(trip) < (60 / sample_rate):  # Skip short trips
        return None

    # Map match (in-place operation)
    match = map_match_valhalla(trip, sample_rate=sample_rate)

    if match is None:
        return None

    trip_distance = trip.speed.sum() * sample_rate
    if trip_distance < 200:  # Skip short trips
        return None

    return trip  # Valid trip

def form_trips(df:pd.DataFrame, boundaries:list[int], sample_rate:float=2.0, max_workers:int=6):
    """Forms trips using multi-threading for better performance."""
    trips = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_trip, df, start, end, sample_rate)
                   for start, end in zip(boundaries, boundaries[1:])]

        for future in tqdm(as_completed(futures), total=len(futures), desc='Map Matching'):
            result = future.result()
            if result is not None:
                trips.append(result)

    return trips

def get_trip_head(trip:pd.DataFrame, sample_rate=2.0, embedding_bins=None):
    departure = trip.iloc[0]['datetime']
    departure = (departure.hour*60 + departure.minute)//5 # 5 minute intervalls
    trip_distance = trip.speed.sum()*sample_rate
    trip_time = trip.iloc[-1]['datetime'] - trip.iloc[0]['datetime']
    trip_time = trip_time.total_seconds()
    sample_points = len(trip)
    avg_dis = trip_distance / sample_points
    avg_speed = trip_distance / trip_time
    sid = eid = 0 # dummy values
    if embedding_bins:
        sid, eid, _ = bin_embedding([trip], bins=embedding_bins)

    fuel_NaN = trip['FuelRate'].max() == -1
    return departure, trip_distance, trip_time, sample_points, avg_dis, avg_speed, sid, eid, fuel_NaN


class trip_normalizer:
    def __init__(self, trips:list[pd.DataFrame], columns=['lat', 'lon', 'FuelRate']):
        self.columns = columns
        *_, self.bins = bin_embedding(trips)

        temp = pd.concat(trips, ignore_index=True).loc[:,columns]
        self.trip_mean = temp.mean().to_numpy()
        self.trip_std  = temp.std().to_numpy()
        del temp
        
        trip_heads = [get_trip_head(trip) for trip in trips]
        trip_heads = np.array(trip_heads)
        self.head_mean = trip_heads.mean(axis=0)
        self.head_std  = trip_heads.std(axis=0)
        classes_index = [0,6,7,8] # override normalization for classes
        self.head_mean[classes_index] = 0
        self.head_std[classes_index]  = 1
        self.head_std[self.head_std==0] = 1 # handle edge case

    def trips2numpy(self, trips:list[pd.DataFrame], nan_tolerance:float=0.05):
        # handle NaNs, constant fill if too many, else forward fill
        for trip in trips:
            for col in trip.columns:
                if trip.loc[:, col].isna().mean() > nan_tolerance:
                    trip.loc[:, col] = -1
                else:
                    trip.loc[:, col] = trip.loc[:, col].ffill().bfill()

        # heads
        sid, eid, _ = bin_embedding(trips, bins=self.bins)
        trip_heads = [get_trip_head(trip) for trip in trips]
        trip_heads = np.array(trip_heads)
        trip_heads[:,6] = sid
        trip_heads[:,7] = eid

        # trips
        trips = [trip.loc[:,self.columns].to_numpy() for trip in trips] # to numpy
        trips = [resample_trajectory(t, length=200) for t in trips]
        trip_arrays = np.ndarray((len(trips), *trips[0].shape))
        for i, trip in enumerate(trips):
            trip_arrays[i] = trip

        # Normalize
        trip_arrays = (trip_arrays-self.trip_mean) / self.trip_std
        trip_heads  = (trip_heads -self.head_mean) / self.head_std
        return trip_arrays, trip_heads

    def __call__(self, trips:list[pd.DataFrame]):
        return self.trips2numpy(trips)

    def numpy2trips(self, trip_arrays:np.ndarray, trip_heads:np.array) -> list[pd.DataFrame]:
        if type(trip_arrays) == torch.Tensor: # in case of tensor data
            trip_arrays = np.asarray(trip_arrays)
            trip_arrays = np.swapaxes(trip_arrays, 1, 2)
            trip_heads = np.asarray(trip_heads)

        if len(trip_arrays.shape) == 2: # add batch dim if none
            trip_arrays = trip_arrays[np.newaxis,:,:]
            trip_heads = trip_heads[np.newaxis,:]
        
        # De-normalize
        trip_arrays = trip_arrays*self.trip_std + self.trip_mean
        trip_heads  = trip_heads *self.head_std + self.head_mean

        # Resample number of points
        lengths = trip_heads[:,3].astype(int)
        trips = [resample_trajectory(trip, length) for trip, length in zip(trip_arrays, lengths)]
        trips = [pd.DataFrame(trip, columns=self.columns) for trip in trips]
        return trips
        
def preprocess(df:pd.DataFrame, file_name:str=None, max_speed:float=100, del_df:bool=False, normalizer:'trip_normalizer'=None):
    'max_speed: max distance between two subsequent points of a trip'
    # add time column
    if 'datetime' not in df.columns:
        df = add_datetime_column(df)
    df = df[(df["datetime"]<datetime(2016,1,1)) & (df["datetime"]>datetime(2010,1,1))] # remove invalid dates
    
    # add speed column
    df = add_speed_column(df)
    
    # split into trip when speed or time is greater than X
    out_of_time = (df['datetime']-df['datetime'].shift(1).bfill()) > timedelta(minutes=30)
    out_of_touch = df['speed'] > max_speed # 100 meters/sec
    boundaries, *_ = np.nonzero(out_of_time | out_of_touch)
    df.loc[boundaries, 'speed'] = 0 # new trips start with 0 speed (no teleporting from last trip)

    # split long trips
    boundaries = np.hstack([0, boundaries, len(df)])
    boundaries = interpolate_indices(boundaries, max_length=200)
    
    # compute heads
    trips = form_trips(df, boundaries)
    if del_df:
        del df # Free RAM

    # Turn trips into fixed length
    if not normalizer:
        normalizer = trip_normalizer(trips)
    trip_arrays, trip_heads = normalizer(trips)
    
    # write and return
    data = trip_arrays, trip_heads, normalizer
    if file_name:
        save_pickle(data, file_name)
    return data

def save_pickle(data, file_name:str):
    'Saves data as pickle. Overwrites the previous file!'
    with open(file_name, 'wb') as file:
        file.seek(0)
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.truncate()

def do_preprocess(cfg):
    paths = cfg.data.datapaths
    # paths = paths[:len(paths)//2] # cut data to prevent running oom. TODO build normalizer first to allow data streaming
    df = open_data(paths)
    og_length = len(df)
    trip_arrays, trip_heads, normalizer = preprocess(df, file_name=cfg.data.preprocessed_datapath, del_df=True)
    print(f'Total: {og_length}, After processing: {sum(trip_heads[:,3]*normalizer.head_std[3]+normalizer.head_mean[3])}')
    return trip_arrays, trip_heads, normalizer
"""
Lets face it: it is called utils because we don't know what to call it. Anything goes in here
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import functools
from numpy import sin, asin, cos
from itertools import islice
from busbus.dataloader import open_data

def no_return(func):
    "Decorator that removes unwanted returns in CLI"
    @functools.wraps(func)
    def wrapper_no_return(*args, **kwargs):
        func(*args, **kwargs)
    return wrapper_no_return

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    'Haversine formula. Returns in meters. Use haversine package for higher accuracy'
    r = 6378.137 * 1000 # earth radius in meter [Moritz, H. (1980). Geodetic Reference System 1980]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2-lat1)/2.0)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2.0)**2
    return 2 * r * asin(a**.5)

def check_fuelrate_NaNs(path:str):
    """
    Checks pickle datapath for missing values and sums up how many consecutive missing values there are.
    
    Conclusion from previous run: FuelRate is missing for full trips. 
    We should use a constant fill and then condition the model on this
    """
    df = pd.read_pickle(path)
    return df.loc[df.FuelRate.isna()].index.diff().value_counts()

def plot_on_map(*dfs, mode:str='line', max_trips:int=100, name:str=None, show:bool=True, hoverdata:str='speed'):
    """
    Plots GPS data on map.
    dfs: dataframes containing one trip each
    mode: 
        'line': convential line plot.
        'scatter': Randomly sized, transparent points, dyed by FuelRate. This stochastically gives bigger and less transparent points in dense areas.
        'density': density weighted by FuelRate data, due to its correlation with translation it is very uniform/uninformative.
    max_trips: limit how many trips are rendered to prevent oom
    name: if present, it will save a png image
    
    
    example usage:
    plot_on_map(df1, df2, mode='density', name=dir/file)
    """
    
    dfs = [df.assign(trip=i) for i, df in enumerate(islice(dfs, max_trips))]
    df = pd.concat(dfs, axis=0)
    if 'FuelRate' in df.columns: # dirty ad hoc sanitation
        df[df.loc[:,'FuelRate'] > 100] = 0
    
    kwargs = {
        'lat':'lat', 
        'lon':'lon', 
        'hover_data':hoverdata, 
        'center':{'lat':df['lat'].mean(), 'lon':df['lon'].mean()},
        'zoom':11, 
        'height':600, 
        'map_style':'open-street-map', 
        'hover_data':[hoverdata]
    }
    
    func = {
        'line':px.line_map,
        'scatter':px.scatter_map,
        'density':px.density_map,
    }[mode]

    update={
        'line':{'color':'trip'},
        'scatter':{
            'color':hoverdata,
            'opacity':len(df)**-.25,
            'size':np.random.rand(len(df))**2,
            'color_continuous_scale':'ylorbr',
        },
        'density':{'z':hoverdata, 'radius':10}
    }[mode]
    kwargs.update(update)
    
    fig = func(df, **kwargs)
    
    if mode == 'line':
        fig.update_layout(margin={v:0 for v in 'rtlb'}, showlegend=False)
    else:
        fig.update_layout(margin={'r':120, 't':0, 'l':0, 'b':10})

    if name:
        plt.savefig(name+'.png');
    [plt.close, fig.show][show]();
    
def hist(*dfs:pd.DataFrame, name:str='histograms.png', legend:list=['Original', 'Generated'], bins:int=30, q:float=.999) -> None:
    'Overlaid histograms for all intersecting columns of multiple dataframes. q sets the fraction of data to include.'

    # Get intersecting columns
    cols = sorted(set.intersection(*map(lambda x: set(x.columns), dfs)))
    length = min(map(len, dfs))
    fig, (axes, *_) = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4), squeeze=False)

    # Plot each column's histograms in its corresponding subplot
    for ax, col in zip(axes, cols):
        # First find the data we need to plot and what range it is in
        col_range = {'min':float('inf'), 'max':float('-inf')}
        col_data = []
        for df in dfs:
            d = df.iloc[:length][col]
            d = d[d < d.quantile(q=q)]
            col_range['min'] = min(col_range['min'], d.min())
            col_range['max'] = max(col_range['max'], d.max())
            col_data.append(d)

        # Plot the data
        colors = [c for c in 'kbgrycm'] # TODO: make this dynamic
        for data, label, color in zip(col_data, legend, colors):
            ax.hist(data, bins=bins, range=col_range.values(), alpha=0.5, label=label, color=color)

        # Set plot properties
        ax.set_xlim(col_range.values())
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()

    plt.tight_layout()
    if name:
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_traj(*gen_traj, scale:float=2, name:str=None, show:bool=True):
    # make plot
    fig = plt.figure(figsize=(8,8));
    for traj in gen_traj:
        y,x,*_ = zip(*traj.to_numpy())
        plt.plot(x,y,color='blue',alpha=0.1)

    # brute force mean and std values
    cat=pd.concat(gen_traj, axis=0) 
    y_mean, x_mean, *_ = cat.mean(axis=0)
    y_std, x_std, *_ = cat.std(axis=0)
    
    # update layout
    ax, *_ = fig.axes
    ax.set_ylim([y_mean-y_std*scale, y_mean+y_std*scale])
    ax.set_xlim([x_mean-x_std*scale, x_mean+x_std*scale])
    plt.tight_layout()
    if name:
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
    [plt.close, plt.show][show]();

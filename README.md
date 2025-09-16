# BusBus - Generating CAN-bus bus data with diffusion
This project is based on difftraj which is made by [Yasoz](https://github.com/Yasoz) but has been modified to include vehicle fuel rate

## Installation
#### Quick guide:
```bash
git clone https://github.com/Jesper-H/BusBus.git
cd BusBus
conda create python=3.10 -n busbus
conda activate busbus
conda install pip
pip install -r requirements.txt
```
Note that Windows users need to remove triton from requirements.txt or install [this windows fork](https://github.com/woct0rdho/triton-windows)

## Usage
#### Download and run Valhalla
This will download Swedens map, you can find other maps [here](https://download.geofabrik.de/)
```bash
cd valhalla
mkdir custom_files
wget -O custom_files/sweden-latest.osm.pbf https://download.geofabrik.de/europe/sweden-latest.osm.pbf
docker pull ghcr.io/nilsnolde/docker-valhalla/valhalla:latest
docker run -dt --name valhalla_gis-ops -p 8002:8002 -v $PWD/custom_files:/custom_files ghcr.io/nilsnolde/docker-valhalla/valhalla:latest
```
#### Prepare data
```
TODO
```
CLI not finished. See main.ipynb for how code can be used instead
#### Train model
```
TODO
```
#### Infer new data
```
python busbus\main.py inference path/to/the/model 1 --save_path=output/dir/file_name.h5
```
#### Visualize
```
TODO
```
# Pyrocb
### Authors
Creator: Dr. Laura Thapa
Maintainer: Jared Goldman (jaredgoldman@ucla.edu)

### Description
A library to create features of wildfire Fire radiative Energy (FRE) and Fire Radiative Power (FRP) prediction from VIIRS, HRRR, MTBS Boundary, population, response, and other data sources. The code will generate features from this data in the form of common fire weather indicies and metadata statistics from the datasets. Additionally, it will plot observed FRE from VIIRS observations of past fires over the MTBS boundaries. 

### Data Requirements
Currently uses data from a local server but will be updated to dynamically load datasets.

## Setup
```
conda env create -f environment.yml
conda activate pyrocb
python setup.py install
```

## Polygons
Run the following command `python src/polygons.py` which will save the polygons into the `outputs/plots` directory.

## Features
Run `python src/data/features/load_all_features.py`
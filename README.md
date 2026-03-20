# Pyrocb
### Authors
Creator: Dr. Laura Thapa
Maintainer: Jared Goldman (jaredgoldman@ucla.edu)

### Description
A library to create features of wildfire Fire radiative Energy (FRE) and Fire Radiative Power (FRP) prediction from VIIRS, HRRR, MTBS Boundary, population, response, and other data sources. The code will generate features from this data in the form of common fire weather indicies and metadata statistics from the datasets. Additionally, it will plot observed FRE from VIIRS observations of past fires over the MTBS boundaries. 

### Data Requirements
Currently uses data from a local server but will be updated to dynamically load datasets.

## Run Data Science Code
## Setup
from root of repo:
```
conda create -n pyrocb -y python=3.10
conda activate pyrocb
python -m pip install -e . 
conda install -c conda-forge -y cartopy geopandas pandas numpy scikit-learn scipy eofs
```

1. Copy the dataset `cleaned_data.csv` into your `pyrocb/outputs/features/` directory (https://drive.google.com/file/d/1wtl44_bBg1ay7D90NcVuvxzacFiqyjWp/view?usp=drive_link)
2. Copy the dataset `cp_poly.gpkg` into your `pyrocb/src/data/` directory (https://drive.google.com/file/d/1_lXk51jk2G-dTSl0cCSgJgMhC5vKmDP8/view?usp=sharing)

All plots will be saved to the `outputs/plots/data_science/` directory

### Generate the eofs
> cd src
> python bndrywise_eof.py

### Train the Ordinary Least Squares Model
> python ml_training.py --pred --plot-dir data_science/models -d cleaned_data.csv  --plot-dir data_science/mode --model ols --name ols_pred1 --pred_days 1
> python ml_training.py --pred --plot-dir data_science/models -d cleaned_data.csv  --plot-dir data_science/mode --model ols --name ols_pred2 --pred_days 2

### Train Random Forest Model
> python ml_training.py --pred --plot-dir data_science/models -d cleaned_data.csv --plot-dir data_science/mode --model rf --name rf_pred1 --pred_days 1
> python ml_training.py --pred --plot-dir data_science/models -d cleaned_data.csv --plot-dir data_science/mode --model rf --name rf_pred2 --pred_days 2
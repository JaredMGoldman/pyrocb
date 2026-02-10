## Background - review paper
* Smoke data from HRRRsmoke data to predict next day smoke data versus persistence

## Process
1. Create Fire Polygons from VIIRS data
    - (v1) Cached data
    - (v2) recent VIIRS
    - (v3) live VIIRS

2. Generate ML Training data

3. Train models

4. Save models and evaluate over recent data

## Research Objectives
* Run Research Code
* Move/rewrite code & data to new repo
* Refactor code to be more dynamic
    * process updated datasets
    * persist data locally or through environment variables

## Research Progress
* 1-12-26 
    * Reviewed repository
        - polygon generation code
        - ml training with derived artifacts
    * tracked artifact generation procedures
        - viirs & sit data aggregation
        - multiyear training dataset aggregation
        - polygons
    * created script to track down files
    * reviewed Thalpa et al research paper

* 1-22-26
    * created polygons from given data
    * trained ml model and evaluated feature importances
    * loaded features
    * refactored code
    * TODO:
        * unify feature creation script
        * load data dynamically (when applicable)
            * consult with Laura on data access creds
        * create a data loading pipeline
        * create a polygon generation pipeline
        * create a ml training/eval pipeline
        * Generate a number of plotting scripts
        * determine appropriate propogation of FRP estimates
     

## Dry Run ideas
* given lat and lon, date of fire ignition and ignition date => create a forecast of FRP growth
    * pull viirs data and HRRR
    * look back one day

* Send email to SDS state about reprocessed RAVE data availability and time frame
    * https://sites.google.com/view/rave-emission/products?authuser=0
        * fangjun.li@sdstate.edu
        * xiaoyang.zhang@sdsstate.edu 

* confirm that lat/lon assigning in xarrays works as intended, mapping x-vals to lats and y-vals to lons

## Data Info
### USA & Canada
1. FIRMS
    - USA & Canada
2. HRRR
    - HRRR is USA Only
    - Canadian Product: https://www.pivotalweather.com/model.php?m=hrdps
3. MODIS
    - US & Canada
4. ESI
    - Worldwide
5. SMOPS
    - data available until 2024...
    - Worldwide
6. RAVE
    - check in with developers at sds State

### Only USA
4. GridMET
    - CONUS Only
6. NCAR FMC
    - missing range from (Oct 2021 - April 2023)
    - CONUS & AK
9. LANDFIRE: Only Elevation and Loading...
    - USA

# Client Checklist
## Built clients
1. FIRMS
    - USA & Canada
2. HRRR
    - HRRR is USA Only
    - Canadian Product: https://www.pivotalweather.com/model.php?m=hrdps
3. MODIS
    - US & Canada
4. GridMET
    - CONUS Only
5. ESI
    - Worldwide
6. NCAR FMC
    - missing range from (Oct 2021 - April 2023)
    - CONUS & AK
7. SMOPS
    - data available until 2024...
    - Worldwide
8. RAVE
    - check in with developers at sds State
9. LANDFIRE: Only Elevation and Loading...
    - USA

## Clients To Build
1. Resources
    - Data is available through portal but tedious and manual process of clicking through screens and selecting specific division/agency
    - need to make a tool that automatically extracts data
    - only saw up to 2024
    - worth checking with Laura
    - Maybe this website? https://inciweb.wildfire.gov/
2. PWS
    - only static map from 2021, assumed static in OG paper

## Next Steps
* which include Canada?
1. Generate incident dataset for each fire based on combo of FIRMS data & MTBS. 
    - Save as a csv
    - find ignition location for newer fires maybe from incident descriptions
2. Create a dashboard with stats about each fire and maps of different layers
3. Train basic ML model and evaluate over dry run + fires
    - Expand model architectures and compare performances
    - feature importance weighting
4. Generate evaluation metrics

## Ideas
* Create statistical model to interpolate SMOPS & FMC data from atmospheric vars
    * paper-worthy especially when comparing it to fire predicatibility performance versus ground truth
        * i.e. make a model and evaluate performance with interpolated vs real data

* Generate new PWC maps... Use old one for now but seems like a weak point
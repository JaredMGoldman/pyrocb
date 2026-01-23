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
     
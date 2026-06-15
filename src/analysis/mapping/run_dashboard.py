# run_dashboard.py
import time
import pandas as pd

from analysis.pft_gen_parallel import pull_data, GFSClient, GFS, group_client_dses, calc_pfts, calc_soundings, parse_to_dataframe
import analysis.mapping.config as config
from analysis.mapping.veda_pft_pipeline import VedaPftLandmaskedPipeline
from analysis.mapping.map_server import MapServer 

# run_dashboard.py updates[cite: 3]
def run_pipeline(pft_dataframe):
    data_pipeline = VedaPftLandmaskedPipeline()
    
    # 1. Fetch Today's Fires[cite: 1]
    print("Fetching active fire perimeters...")
    fire_geojson = data_pipeline.fetch_veda_fires(days_back=1)
    
    # 2. Compile Map with PFT Mesh and Fire Layers[cite: 1]
    print("Compiling spatiotemporal map...")
    data_pipeline.compile_integrated_map(
        geojson_data=fire_geojson,
        pft_df=pft_dataframe,
        output_html=config.OUTPUT_HTML
    )
    
    # 3. Server Deployment[cite: 3]
    server = MapServer(port=config.SERVER_PORT, html_filename=config.OUTPUT_HTML)
    server.start()
    
    try:
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[-] Shutting down network server sockets cleanly...")
        server.stop()

if __name__ == "__main__":
    print("=== STEP 1: Ingesting GFS Atmospheric Arrays ===")
    dtime = config.now_date
    lat = config.lats
    lon = config.lons
    fxx_range = config.fxx_range
    clients = config.clients
    fx_names = config.fx_names
    max_workers = config.max_workers

    # Execute your parallel processing pipeline
    dses = pull_data(dtime, lat, lon, fxx_range, clients, max_workers)
    client_dses = group_client_dses(dses, fx_names)
    soundings = calc_soundings(client_dses, max_workers)
    pfts = calc_pfts(soundings, max_workers)
    
    # Generate your active predictive dataframe
    print("[+] Meteorological matrix calculation complete.")
    df_pfts_calculated = parse_to_dataframe(pfts)
    
    # Hand the dataset directly to your dashboard tracking maps
    run_pipeline(df_pfts_calculated)
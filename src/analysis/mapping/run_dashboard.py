import os
import shutil

from analysis.pft_gen_parallel import pull_data, group_client_dses, \
                                        calc_pfts, calc_soundings, \
                                        parse_to_dataframe, CACHE_BASE_DIR
import analysis.mapping.config as config
from analysis.mapping.copy_util import upload_simplified

# run_dashboard.py updates[cite: 3]
def run_pipeline():
    print("=== STEP 1: Ingesting Atmospheric Data ===")
    dtime = config.now_date
    lat = config.lats
    lon = config.lons
    fxx_range = config.fxx_range
    fxx_freq = config.fxx_freq
    clients = config.clients
    fx_names = config.fx_names
    max_workers = config.max_workers
    cmap_name = config.cmap_name

    fire_fetch_class = config.active_fire_class

    csv_fname = 'fire_pipeline_manifest.csv'
    csv_current_dir = os.path.join(CACHE_BASE_DIR, 'active_fires', 'current')
    csv_today_dir = os.path.join(CACHE_BASE_DIR, 'active_fires', dtime.replace('-','_'))
    os.makedirs(csv_today_dir, exist_ok=True)

    # 1. Fetch Today's Fires[cite: 1]
    print("Fetching active fire perimeters...")
    data_pipeline = fire_fetch_class()
    fire_geojson = data_pipeline.fetch_fires(days_back=1, csv_path = os.path.join(csv_today_dir, csv_fname))
    os.makedirs(csv_today_dir, exist_ok=True)
    shutil.copy(os.path.join(csv_today_dir, csv_fname), os.path.join(csv_current_dir, csv_fname))
    print(f"copied current fires to {os.path.join(csv_current_dir, csv_fname)}")

    # Execute your parallel processing pipeline
    dses = pull_data(dtime, lat, lon, fxx_range, clients, max_workers, fxx_freq)
    client_dses = group_client_dses(dses, fx_names)
    soundings = calc_soundings(client_dses, max_workers)
    pfts = calc_pfts(soundings, max_workers)
    
    # Generate your active predictive dataframe
    print("[+] Meteorological matrix calculation complete.")
    df_pfts_calculated = parse_to_dataframe(pfts)
    
    # Hand the dataset directly to your dashboard tracking maps
    df_pfts_calculated.to_csv(os.path.join(csv_current_dir, 'pft_data.csv'))
    df_pfts_calculated.to_csv(os.path.join(csv_today_dir, 'pft_data.csv'))
    
    # 2. Compile Map with PFT Mesh and Fire Layers[cite: 1]
    print("Compiling spatiotemporal map...")
    data_pipeline.compile_integrated_map(
        geojson_data=fire_geojson,
        pft_df=df_pfts_calculated,
        output_html=config.OUTPUT_HTML,
        cmap_name=cmap_name
    )

    html_fname = config.OUTPUT_HTML.split(os.path.sep)[-1]
    upload_simplified(config.OUTPUT_HTML, os.path.join(config.REMOTE_DIR, html_fname),
                      hostname = config.HOSTNAME, username=config.USERNAME)
    if not config.DEBUG_MODE:
        upload_simplified(config.OUTPUT_HTML, os.path.join(config.REMOTE_DIR, "latest.html"),
                        hostname = config.HOSTNAME, username=config.USERNAME)

if __name__ == "__main__":
    run_pipeline()
import os
import base64
import datetime
from pathlib import Path
import pandas as pd
import folium
from shapely import wkt
from shapely.geometry import mapping

import analysis.realtime_pft.realtime_pft_config as pft_config
from analysis.mapping_utils import encode_image_to_base64, add_local_png_favicon
from utils.constants import CACHE_BASE_DIR


def generate_interactive_pft_map(
    manifest_csv: str, 
    pft_csv_path: str, 
    plots_dir: str
) -> Path:
    """
    Generates a Folium interactive map with base layer toggles and a consolidated 
    'Active Fires' polygon layer.
    """
    manifest_path = Path(manifest_csv)
    pft_csv = Path(pft_csv_path)
    plots_folder = Path(plots_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    today_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    # 1. Base Map: Esri World Imagery
    esri_satellite_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    esri_attribution = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"

    m = folium.Map(
        location=[39.8283, -98.5795], 
        zoom_start=4,
        tiles=None  # Leave blank so TileLayer controls naming cleanly
    )

    # Base Layer 1: Esri Satellite
    esri_satellite_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    esri_attribution = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"

    folium.TileLayer(
        tiles=esri_satellite_url,
        attr=esri_attribution,
        name="Esri Satellite (Base)",
        overlay=False,
        control=True
    ).add_to(m)

    # Base Layer 2: OpenStreetMap
    folium.TileLayer(
        tiles='openstreetmap', 
        name='OpenStreetMap (Base)',
        overlay=False,
        control=True
    ).add_to(m)

    # 2. Near-Real-Time NASA GIBS IR Cloud Cover Overlay (EPSG:3857 TileLayer)
    # Uses yesterday/today date string to avoid requesting future/unprocessed satellite passes
    today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    gibs_cloud_url = (
        f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"GOES-East_ABI_Band13_Clean_Infrared/default/{today_str}/"
        f"GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.png"
    )

    goes_clouds = folium.TileLayer(
        tiles=gibs_cloud_url,
        attr="NASA EOSDIS GIBS / NOAA GOES-East",
        name="GOES IR Cloud Cover (Real-Time)",
        overlay=True,
        opacity=0.5,
        show=False
    )
    goes_clouds.add_to(m)

    esri_boundaries_url = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
    
    borders_layer = folium.TileLayer(
        tiles=esri_boundaries_url,
        attr="Tiles &copy; Esri &mdash; Boundaries & Places",
        name="State/Country Borders",
        overlay=True,
        opacity=0.8,
        show=True
    )
    borders_layer.add_to(m)

    # 4. Single FeatureGroup for Active Fires
    fires_layer = folium.FeatureGroup(name="Active Fires", show=True)

    added_fires = 0
    for _, row in df_manifest.iterrows():
        fire_id = row.get('fire_index_id', row.get('fire_id'))
        # Fixed: Match 'fire_name' column from manifest
        fire_name = row.get('fire_name', row.get('name', fire_id))
        wkt_str = row.get('wkt_geometry')

        if pd.isna(wkt_str):
            continue

        try:
            geom = wkt.loads(wkt_str)
            geojson_geom = mapping(geom)
        except Exception as e:
            print(f"[-] Failed to parse WKT for fire {fire_id}: {e}")
            continue

        # Look up corresponding plot image matching plot generator logic
        safe_fire_name = str(fire_name).replace(" ", "_").replace("/", "_")
        plot_path = plots_folder / f"pft_plot_{safe_fire_name}.jpg"

        if plot_path.exists():
            b64_img = encode_image_to_base64(plot_path)
            popup_html = f"""
            <div style="width: 320px; font-family: Arial, sans-serif;">
                <h4 style="margin: 0 0 8px 0; color: #b30000;">{fire_name}</h4>
                <img src="data:image/jpeg;base64,{b64_img}" style="width: 100%; border-radius: 4px;" />
            </div>
            """
        else:
            popup_html = f"""
            <div style="width: 200px; font-family: Arial, sans-serif;">
                <h4 style="margin: 0 0 5px 0;">{fire_name}</h4>
                <p style="color: #666; font-size: 11px;">No PFT plot available for this footprint.</p>
            </div>
            """

        popup = folium.Popup(popup_html, max_width=350)
        tooltip = f"{fire_name}"

        geojson_feature = {
            "type": "Feature",
            "geometry": geojson_geom,
            "properties": {"name": fire_name, "id": fire_id}
        }

        # High-visibility polygon styling
        folium.GeoJson(
            geojson_feature,
            style_function=lambda x: {
                'fillColor': '#FFCC00',
                'color': '#FF007F',
                'weight': 3,
                'fillOpacity': 0.6
            },
            highlight_function=lambda x: {
                'fillColor': '#00FFFF',
                'color': '#FFFFFF',
                'weight': 4,
                'fillOpacity': 0.85
            },
            popup=popup,
            tooltip=tooltip
        ).add_to(fires_layer)

        added_fires += 1

    # Attach fire group to map
    fires_layer.add_to(m)

    # SINGLE LayerControl call at the end of map assembly
    folium.LayerControl(collapsed=False).add_to(m)

    m = add_local_png_favicon(m, pft_config.inspyre_icon_path)

    output_dir = Path(CACHE_BASE_DIR) / "pft_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_tag = pft_csv.stem.replace("pft_rrfs_", "") if pft_csv else "latest"
    output_html_path = output_dir / f"realtime_pft_{run_tag}.html"
    
    m.save(str(output_html_path))
    print(f"[+] Interactive map generated with {added_fires} fire overlays: {output_html_path}")
    return output_html_path


if __name__ == "__main__":
    from analysis.realtime_pft.active_pft_maker import run_parallel_pft_from_netcdf, \
                                                        generate_realtime_pft_plots
    from analysis.mapping.copy_util import upload_simplified

    if not os.path.exists(pft_config.manifest_path):
        raise RuntimeError(f"No manifest found at {pft_config.manifest_path}")

    client = pft_config.pft_oper_client(csv_manifest_path=pft_config.manifest_path)
    grib_paths, cycle_date, cycle_hour = client.download_latest_prslev_data(fxx_hours=range(0, pft_config.fxx_range + 1,
                                                                                            pft_config.fxx_interval))
    summary_csv = client.process_fires_in_parallel(grib_paths, max_workers=pft_config.max_workers)

    # 2. Run PFT processor on NetCDF files in parallel (no dataset merging)
    out_csv = run_parallel_pft_from_netcdf(
        summary_csv_path=summary_csv,
        manifest_path = pft_config.manifest_path,
        cycle_date=cycle_date,
        cycle_hour=cycle_hour,
        max_workers=pft_config.max_workers
    )
    
    generate_realtime_pft_plots(out_csv, max_workers=pft_config.max_workers)

    csv_files = sorted(pft_config.pft_out_dir.glob("pft_rrfs_*.csv"))
    latest_csv = csv_files[-1]
    run_tag = latest_csv.stem.replace("pft_rrfs_", "")
    plots_dir = Path(CACHE_BASE_DIR) / "pft_plots" / f"run_{run_tag}"

    html_path = generate_interactive_pft_map(
            manifest_csv=pft_config.manifest_path,
            pft_csv_path=latest_csv,
            plots_dir=plots_dir
        )
    html_fname = str(html_path).split(os.path.sep)[-1]
    upload_simplified(html_path, os.path.join(pft_config.REMOTE_DIR, 
                                              html_fname),
                          hostname = pft_config.HOSTNAME, 
                          username = pft_config.USERNAME)
    upload_simplified(html_path, os.path.join(pft_config.REMOTE_DIR, 
                                                  "latest.html"),
                              hostname = pft_config.HOSTNAME, 
                              username = pft_config.USERNAME)
from abc import ABC, abstractmethod
import base64
import io
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import folium
import shapely
from shapely.geometry import shape, mapping, Polygon, Point
from shapely import STRtree
from shapely.ops import nearest_points
import cartopy.io.shapereader as shapereader
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

import analysis.mapping.config as config
from analysis.mapping.timeslider_choropleth_utc import TimeSliderChoropleth as TimeSliderChoroplethUTC


def _worker_render_single_plot_frp(fire_key, frp_csv_path):
    matplotlib.use('Agg')
    try:
        frp_df = pd.read_csv(frp_csv_path)
        fire_sub = frp_df[frp_df['fire_idx'] == fire_key].copy()

        if fire_sub.empty:
            return fire_key, f"<p style='color:gray;'>No FRP forecast data available for {fire_key}.</p>"

        fire_name = fire_sub.iloc[0].fire_name.replace('_', ' ').upper()
        fire_sub['time'] = pd.to_datetime(fire_sub['time'])
        fire_sub = fire_sub.sort_values('time')

        # 1. Reduced figure dimensions & lower DPI (75 vs 150 cuts Base64 payload by ~75%)
        fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=75)
        
        has_plot = False
        if 'fc_hybrid' in fire_sub.columns and not fire_sub['fc_hybrid'].isna().all():
            ax.plot(fire_sub['time'], fire_sub['fc_hybrid'], color='green', linestyle='-', linewidth=1.5, label='Hybrid Blend')
            has_plot = True
        if 'rave_historical' in fire_sub.columns and not fire_sub['rave_historical'].isna().all():
            ax.plot(fire_sub['time'], fire_sub['rave_historical'], color='gray', linestyle='-', linewidth=1.5, label='RAVE Observations')
            has_plot = True
            
        if not has_plot:
            plt.close(fig)
            return fire_key, f"<p style='color:gray;'>FRP data coordinates are empty for {fire_key}.</p>"
            
        ax.set_title(f"FRP Forecast Trend: {fire_name}", fontsize=8, fontweight='bold')
        ax.set_ylabel("Total FRP [MW]", fontsize=7)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=config.plot_freq))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=20)
        plt.tight_layout()
        
        # 2. Save compressed JPEG instead of uncompressed high-DPI PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', pil_kwargs={'quality': 81, 'optimize': True})
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        chart_html = f'<img src="data:image/jpeg;base64,{encoded}" style="max-width:100%; height:auto;">'
        return fire_key, chart_html
        
    except Exception as e:
        return fire_key, f"<p style='color:red;'>Error generating FRP plot: {str(e)}</p>"


def _worker_render_single_plot(fire_key, subset_df, fx_name):
    matplotlib.use('Agg')
    if subset_df.empty:
        return fire_key, "<p style='color:gray;'>No PFT metrics found overlaying or near this footprint.</p>"

    pivoted_df = subset_df.pivot_table(
        index='time', 
        columns='plume_temp', 
        values='value', 
        aggfunc='mean'
    ).sort_index()

    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=75)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    x_labels = pd.to_datetime([pd.to_datetime(t).strftime('%m/%d %H:%M') for t in pivoted_df.index])
    colors = plt.cm.tab10.colors

    for i, temp in enumerate(config.MAX_PLUME_TOP_TS):
        if temp in pivoted_df.columns:
            ax.plot(
                x_labels, 
                pivoted_df[temp].values, 
                color=colors[i % len(colors)], 
                linewidth=1.2, 
                marker='o', 
                markersize=2, 
                label=f"{temp}°C"
            )
    
    ax.set_title(f"PFT {fx_name.upper()} Prediction", color='black', fontsize=8, fontweight='bold')
    ax.set_ylabel("PFT Value (GW)", color='black', fontsize=7)
    ax.set_yscale('log')
    ax.tick_params(colors='black', labelsize=6)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=config.plot_freq))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.grid(True, color='#444444', linestyle='--', alpha=0.5)
    ax.legend(title="Plume Temp", fontsize=5, title_fontsize=6, loc='upper left')

    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    # Compressed JPEG export
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', pil_kwargs={'quality': 81, 'optimize': True})
    plt.close(fig)
    buf.seek(0)
    
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    html_img_tag = f'<img src="data:image/jpeg;base64,{base64_img}" width="420" height="220">'
    return fire_key, html_img_tag


def round_geometry_coords(geom_dict, precision=4):
    """Recursively truncates geometry coordinates to 4 decimal places (~11m accuracy)."""
    if "coordinates" in geom_dict:
        def _round(coords):
            if isinstance(coords[0], (int, float)):
                return [round(c, precision) for c in coords]
            return [_round(c) for c in coords]
        geom_dict["coordinates"] = _round(geom_dict["coordinates"])
    return geom_dict

class FireMapBase(ABC):
    def __init__(self):
        self.unified_land_mask = self._build_static_land_mask()

    @abstractmethod
    def fetch_fires(self, *args, **kwargs):
        pass

    def _build_static_land_mask(self):
        """Builds a unified land mass for US and Canada while masking out major inland water bodies."""
        country_shp = shapereader.natural_earth(
            resolution='50m', category='cultural', name='admin_0_countries'
        )
        country_reader = shapereader.Reader(country_shp)
        land_geoms = [
            country.geometry for country in country_reader.records() 
            if country.attributes['NAME'] in ['United States of America', 'Canada']
        ]
        unified_ground = unary_union(land_geoms)
        return unified_ground

    def export_fire_statistics_csv(self, geojson_data, output_csv="active_fires_summary.csv"):
        """
        Parses compiled fire feature clusters to compile structural statistics,
        converting vector boundaries to Well-Known Text (WKT) geometries.
        """
        if not geojson_data or not geojson_data.get("features"):
            print("[-] Warning: No active fire features available to parse into CSV summary.")
            return None

        records = []
        for feature in geojson_data["features"]:
            props = feature.get("properties", {})
            geom = feature.get("geometry", None)
            
            if not geom:
                continue

            try:
                shapely_geom = shape(geom)
                wkt_geometry = shapely_geom.wkt
                
                fire_id = props.get("fireid", "UNKNOWN")
                peak_area = props.get("farea", 0.0)
                mean_frp = props.get("meanfrp", 0.0)
                
                t_start_raw = props.get("t_start", props.get("t", None))
                t_end_raw = props.get("t_end", props.get("t", None))
                
                duration_days = 0.0
                start_date_str = ""
                end_date_str = ""
                
                if t_start_raw and t_end_raw:
                    dt_start = pd.to_datetime(t_start_raw).tz_localize(None)
                    dt_end = pd.to_datetime(t_end_raw).tz_localize(None)
                    
                    start_date_str = dt_start.strftime("%Y-%m-%d %H:%M:%S")
                    end_date_str = dt_end.strftime("%Y-%m-%d %H:%M:%S")
                    
                    delta = dt_end - dt_start
                    duration_days = round(delta.total_seconds() / 86400.0, 2)
                
                records.append({
                    "fire_id": fire_id,
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "duration_days": duration_days,
                    "peak_area_km2": peak_area,
                    "mean_viirs_frp": mean_frp,
                    "wkt_geometry": wkt_geometry
                })
            except Exception as ex:
                print(f"[-] Bypassed single feature profile calculation: {ex}")
                continue
                
        if records:
            df = pd.DataFrame(records)
            df = df.sort_values(by="mean_viirs_frp", ascending=False)
            df.to_csv(output_csv, index=False)
            print(f"[+] Active fire statistical inventory saved successfully: '{output_csv}'")
            return df
        return None


    def _generate_html_legend(self, levels, cmap_name='viridis_r'):
        """Generates a responsive floating CSS legend for discrete level steps."""
        cmap = plt.get_cmap(cmap_name)
        
        # Sample colors at step midpoints / normalized levels
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N)
        # Get a representative color for each discrete level block
        midpoints = [(levels[i] + levels[i+1]) / 2 for i in range(len(levels)-1)]
        hex_colors = [mcolors.to_hex(cmap(norm(m))) for m in midpoints]
        
        # Build CSS linear-gradient string with discrete stops
        stops = []
        n_blocks = len(hex_colors)
        for i, color in enumerate(hex_colors):
            p_start = (i / n_blocks) * 100
            p_end = ((i + 1) / n_blocks) * 100
            stops.append(f"{color} {p_start:.1f}%, {color} {p_end:.1f}%")
        gradient_css = ", ".join(stops)
        
        # Select key level labels to display across the legend bottom
        display_ticks = [levels[0], levels[len(levels)//4], levels[len(levels)//2], levels[3*len(levels)//4], levels[-1]]
        
        legend_html = f"""
        <div style="
            position: fixed; 
            bottom: 50px; left: 50px; width: 340px; height: 85px; 
            z-index:9999; font-size:12px; font-family: 'Arial', sans-serif;
            background-color: rgba(20, 20, 20, 0.85);
            color: #ffffff; border-radius: 6px; padding: 12px;
            box-shadow: 0 0 15px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px; color: #ffaa00;">
                PyroCB Firepower Threshold (PFT) [GW]
            </div>
            <div style="
                width: 100%; height: 14px; 
                background: linear-gradient(to right, {gradient_css});
                border-radius: 3px; margin-bottom: 6px;
                border: 1px solid #555;
            "></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #cccccc;">
                {''.join([f'<span>{t}</span>' for t in display_ticks])}
            </div>
        </div>
        """
        return legend_html

    def compile_integrated_map(self, pft_path, manifest_path,
                           output_html="weekly_fire_map.html", 
                           cmap_name='autumn', frp_csv_path='cache'):
        """
        Builds a spatiotemporal hourly-precision grid of PFT values over land mass 
        with a functional dropdown selector to dynamically repaint plume temperature values.
        """
        print("Compiling spatiotemporal map...")
        fire_manifest_df = pd.read_csv(manifest_path)
        pft_df = pd.read_csv(pft_path)

        m = folium.Map(location=[45, -100], zoom_start=4, tiles="CartoDB positron")
        
        if pft_df.empty:
            print("[-] Warning: PFT DataFrame is empty. Skipping spatiotemporal mesh processing.")
            return m

        extent = config.bounds 
        grid_res = config.grid_res         
        
        lon_edges = np.linspace(extent[0], extent[2], grid_res[1] + 1)
        lat_edges = np.linspace(extent[1], extent[3], grid_res[0] + 1)
        
        grid_features = []
        cell_idx = 0
        cell_centroids = []

        for i in range(grid_res[0]):
            for j in range(grid_res[1]):
                w, e = lon_edges[j], lon_edges[j+1]
                s, n = lat_edges[i], lat_edges[i+1]
                cell_poly = Polygon([(w, s), (e, s), (e, n), (w, n)])
                
                if self.unified_land_mask.intersects(cell_poly):
                    land_cell = self.unified_land_mask.intersection(cell_poly)
                    if not land_cell.is_empty:
                        rounded_geom = round_geometry_coords(mapping(land_cell), precision=4)
                        feature = {
                            "type": "Feature",
                            "id": str(cell_idx),
                            "geometry": rounded_geom,
                            "properties": {}
                        }
                        grid_features.append(feature)
                        cell_centroids.append((cell_poly.centroid.x, cell_poly.centroid.y, cell_idx))
                        cell_idx += 1

        geo_path_collection = {"type": "FeatureCollection", "features": grid_features}

        # 1. Setup color scale and levels
        levels = [5, 7.5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')

        clean_pft = pft_df[np.isfinite(pft_df['value'])].copy()
        plume_levels = sorted(clean_pft['plume_temp'].unique())

        # 2. Build style dictionaries for ALL plume temperatures
        plume_style_dicts = {}

        for temp in plume_levels:
            pft_temp_df = clean_pft[clean_pft['plume_temp'] == temp]
            timestamps = sorted(pft_temp_df['time'].unique())

            # Keep dictionary structure lean (no pre-populating zero values)
            style_dict = {}

            for ts in timestamps:
                ts_group = pft_temp_df[pft_temp_df['time'] == ts]
                if len(ts_group) < 4:
                    continue

                ts_naive = pd.to_datetime(ts, utc=True).tz_localize(None)
                unix_sec = str(int(ts_naive.timestamp()))

                points = ts_group[['lon', 'lat']].values
                values = ts_group['value'].values

                grid_coords = [(c[0], c[1]) for c in cell_centroids]
                interpolated_values = griddata(points, values, grid_coords, method='linear')

                for val, cell_info in zip(interpolated_values, cell_centroids):
                    # SPARSE ENCODING: Skip NaNs and values below min threshold entirely
                    if np.isnan(val) or val < levels[0]:
                        continue

                    c_idx = str(cell_info[2])
                    rgba = cmap(norm(val))
                    hex_color = mcolors.to_hex(rgba)
                    
                    if c_idx not in style_dict:
                        style_dict[c_idx] = {}

                    style_dict[c_idx][unix_sec] = hex_color

                    # {
                    #     'color': hex_color,
                    #     'opacity': 0.65
                    # }

            plume_style_dicts[str(temp)] = style_dict

        default_temp_str = str(plume_levels[0])

        # Extract ALL timestamps across all plume temperatures for the slider axis
        all_timestamps = set()
        for temp_dict in plume_style_dicts.values():
            for cell_dict in temp_dict.values():
                all_timestamps.update(cell_dict.keys())

        # Build a minimal 1-cell timeline dictionary to pass to Folium (~1 KB)
        sample_cell_id = next(iter(plume_style_dicts[default_temp_str].keys())) if plume_style_dicts[default_temp_str] else "0"
        minimal_styledict = {
            sample_cell_id: {ts: "#000000" for ts in sorted(all_timestamps)}
        }

        # 3. Add TimeSliderChoropleth with the minimal timeline structure
        time_slider = TimeSliderChoroplethUTC(
            data=geo_path_collection,
            styledict=minimal_styledict,
            name="Predictive PFT Forward Mesh Grid",
            stroke_width=0.0,
            date_options="YYYY-MM-DD_HH:mm [UTC]"
        )
        time_slider.add_to(m)

        slider_var_name = time_slider.get_name()

        options_html = "".join([
            f'<option value="{temp}" {"selected" if str(temp)==default_temp_str else ""}>{temp}°C</option>'
            for temp in plume_levels
        ])

        # 4. Inject Control JS & Hydrate Full Grid on Load
        plume_control_html = f"""
        <div id="plume-temp-container" style="
            position: fixed; 
            top: 10px; 
            right: 10px; 
            z-index: 9999; 
            background: #ffffff; 
            padding: 8px 12px; 
            border-radius: 6px; 
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); 
            font-family: Arial, sans-serif; 
            font-size: 12px;
            border: 1px solid #ccc;
        ">
            <label for="plume-temp-select" style="font-weight: bold; margin-right: 5px; color: #333;">Plume Temp:</label>
            <select id="plume-temp-select" style="padding: 3px 6px; border-radius: 4px; border: 1px solid #aaa;" onchange="switchPlumeTemp(this.value)">
                {options_html}
            </select>
        </div>

        <script>
            var masterPlumeStyles = {json.dumps(plume_style_dicts)};

            function switchPlumeTemp(selectedTemp) {{
                var newStyles = masterPlumeStyles[selectedTemp];
                var updateFunc = window["updateTimeSliderStyle_{slider_var_name}"];

                if (typeof updateFunc === "function" && newStyles) {{
                    updateFunc(newStyles);
                }} else {{
                    console.warn("TimeSlider repaint function not found for layer {slider_var_name}");
                }}
            }}

            // Safe initializer: Polls until Leaflet has attached element IDs to the DOM
            function initPlumeMesh() {{
                var updateFunc = window["updateTimeSliderStyle_{slider_var_name}"];
                var pathsReady = document.querySelector("path[id^='{slider_var_name}-feature-']");

                if (typeof updateFunc === "function" && pathsReady) {{
                    switchPlumeTemp("{default_temp_str}");
                }} else {{
                    setTimeout(initPlumeMesh, 50); // Re-check every 50ms until DOM paths are bound
                }}
            }}

            // Trigger initialization
            initPlumeMesh();
        </script>
        """
        m.get_root().html.add_child(folium.Element(plume_control_html))

        # 5. Add Legend
        legend_html_content = self._generate_html_legend(levels, cmap_name=cmap_name)
        m.get_root().html.add_child(folium.Element(legend_html_content))

        # 6. Active Fires Feature Layer
        if fire_manifest_df is not None and not fire_manifest_df.empty:
            pft_points = []
            for lon, lat in tqdm.tqdm(zip(clean_pft['lon'], clean_pft['lat']), desc='formatting STR tree', total=len(clean_pft['lat'])):
                pft_points.append(Point(lon, lat))

            spatial_tree = STRtree(pft_points)

            prepared_tasks = []
            for f_idx in tqdm.tqdm(fire_manifest_df.fire_index_id.values, desc='preprocessing pft subsets'):
                geom = fire_manifest_df[fire_manifest_df.fire_index_id == f_idx]['wkt_geometry'].item()

                if not geom:
                    continue
                
                try:
                    fire_poly = shapely.wkt.loads(geom)
                except Exception:
                    continue
                    
                indices_in_bbox = spatial_tree.query(fire_poly)
                inside_indices = [idx for idx in indices_in_bbox if fire_poly.contains(pft_points[idx])]
                
                if inside_indices:
                    subset_df = clean_pft.iloc[inside_indices].copy()
                else:
                    nearest_geom_idx = spatial_tree.nearest(fire_poly)
                    nearest_point = pft_points[nearest_geom_idx]
                    coordinate_mask = (clean_pft['lon'] == nearest_point.x) & (clean_pft['lat'] == nearest_point.y)
                    subset_df = clean_pft[coordinate_mask].copy()
                    
                prepared_tasks.append((f_idx, subset_df))
                
            fires_layer_group = folium.FeatureGroup(name="Active Fires", show=True)

            print(f"[+] Launching parallel rendering across worker pools for {len(prepared_tasks)} fires...")
            
            pft_chart = {}
            frp_chart = {}
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                futures = [
                    executor.submit(_worker_render_single_plot, fire_key, sub_df, config.fx_names[0])
                    for fire_key, sub_df in prepared_tasks
                ]
                
                for f in tqdm.tqdm(as_completed(futures), total=len(prepared_tasks), desc="PFT Popup Plots"):
                    fire_key, chart_html = f.result()
                    pft_chart[fire_key] = chart_html
                
                futures = [
                    executor.submit(_worker_render_single_plot_frp, fire_key, frp_csv_path) 
                    for fire_key in list(fire_manifest_df.fire_index_id.values)
                ]

                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="FRP Popup Plots"):
                    try:
                        f_key, html_str = future.result()
                        frp_chart[f_key] = html_str
                    except Exception as e:
                        print(f"[-] A worker process thread failed: {e}")

            for f_key in fire_manifest_df.fire_index_id.values:
                this_fire = fire_manifest_df[fire_manifest_df.fire_index_id == f_key]

                f_name = this_fire['name'].item()
                f_id = this_fire['fire_id'].item()
                f_start = this_fire['start_date'].item()
                f_status = this_fire['status'].item()

                pft_html = pft_chart.get(f_key, "<p style='color:gray;'>Error generating plot data.</p>")
                frp_html = frp_chart.get(f_key, "<p style='color:gray;'>Error generating plot data.</p>")

                popup_content = f"""
                <div style="font-family: Arial, sans-serif; font-size: 12px; color: #FFFFFF; background-color: #FFFFFF; padding: 10px; border-radius: 4px; width: 460px;">
                    <h4 style="margin: 0 0 5px 0; color: #e74c3c; border-bottom: 1px solid #444;">{f_name}</h4>
                    <table style="width:100%; margin-bottom: 10px; font-size:11px;">
                        <tr><td><b>Fire ID:</b></td><td>{f_id}</td></tr>
                        <tr><td><b>Start Date:</b></td><td>{f_start}</td></tr>
                        <tr><td><b>Status:</b></td><td>{f_status}</td></tr>
                    </table>
                    <div style="text-align: center;">
                        {pft_html}
                    </div>
                    <div style="text-align: center;">
                        {frp_html}
                    </div>
                </div>
                """
                
                iframe = folium.IFrame(html=popup_content, width=480, height=360)
                custom_popup = folium.Popup(iframe, max_width=500)
                
                feature = {
                    "type": "Feature",
                    "properties": {"name": f_name, "fireid": f_id},
                    "geometry": mapping(shapely.wkt.loads(this_fire['wkt_geometry'].item()))
                }
                
                folium.GeoJson(
                    feature,
                    style_function=lambda x: {
                        'fillColor': '#e74c3c',
                        'color': '#c0392b',
                        'weight': 2,
                        'fillOpacity': 0.6
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'fireid'], 
                        aliases=['Name:', 'ID:']
                    ),
                    embed=False
                ).add_child(custom_popup).add_to(fires_layer_group)

            fires_layer_group.add_to(m)

        # 7. Layer Control
        folium.LayerControl(collapsed=False).add_to(m)

        m.save(output_html)
        print(f"[+] Spatiotemporal dashboard generated with hourly slider and colorbar: '{output_html}'")
        return m
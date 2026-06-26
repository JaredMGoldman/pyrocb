from abc import ABC, abstractmethod
from herbie import paint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import folium
from folium.plugins import TimeSliderChoropleth
from shapely.geometry import shape, mapping, Polygon
import cartopy.io.shapereader as shapereader
from shapely.ops import unary_union

import analysis.mapping.config as config


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
        lakes_shp = shapereader.natural_earth(
            resolution='50m', category='physical', name='lakes'
        )
        lakes_reader = shapereader.Reader(lakes_shp)
        lake_geoms = [lake.geometry for lake in lakes_reader.records()]
        unified_lakes = unary_union(lake_geoms)

        clean_landmass = unified_ground.difference(unified_lakes)
        return clean_landmass

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
                # Safely parse the structural GeoJSON point/polygon using shapely for WKT string conversion
                shapely_geom = shape(geom)
                wkt_geometry = shapely_geom.wkt
                
                # Extract identifiers and spatial properties safely
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

    def interpolate_pft_field(self, pft_df):
        """Interpolates PFT onto a grid and masks by land."""
        if pft_df.empty: return None, None
        
        # Grid Setup
        extent = config.bounds # Focused NA View
        grid_lons = np.linspace(extent[0], extent[1], 300)
        grid_lats = np.linspace(extent[2], extent[3], 200)
        grid_x, grid_y = np.meshgrid(grid_lons, grid_lats)
        
        # Group by time to create temporal frames
        frames = []
        clean_df = pft_df[np.isfinite(pft_df['value'])]
        vmin, vmax = clean_df['value'].min(), np.percentile(clean_df['value'], 99)
        norm = mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
        cmap = plt.get_cmap('autumn')

        for timestamp, group in clean_df.groupby('time'):
            grid_z = griddata(
                (group['lon'], group['lat']), group['value'], 
                (grid_x, grid_y), method='linear'
            )
            
            # Mask grid points outside land
            for i in range(len(grid_lats)):
                for j in range(len(grid_lons)):
                    if not self.unified_land_mask.contains(shape({"type": "Point", "coordinates": [grid_lons[j], grid_lats[i]]})):
                        grid_z[i, j] = np.nan
            
            # Convert to RGBA
            rgba = cmap(norm(grid_z))
            rgba[np.isnan(grid_z)] = [0, 0, 0, 0] # Transparent ocean
            frames.append({"time": timestamp, "image": np.flipud(rgba)})
            
        return frames, extent

    def _generate_html_legend(self, vmin, vmax, cmap_name='autumn'):
        """Generates a responsive floating CSS gradient legend matching the log-scale mesh color map."""
        cmap = plt.get_cmap(cmap_name)
        samples = np.linspace(0, 1, 6)
        hex_colors = [mcolors.to_hex(cmap(s)) for s in samples]
        ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
        gradient_css = ", ".join(hex_colors)
        
        legend_html = f"""
        <div style="
            position: fixed; 
            bottom: 50px; left: 50px; width: 320px; height: 85px; 
            z-index:9999; font-size:12px; font-family: 'Arial', sans-serif;
            background-color: rgba(20, 20, 20, 0.85);
            color: #ffffff; border-radius: 6px; padding: 12px;
            box-shadow: 0 0 15px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px; color: #ffaa00;">
                PyroCB Firepower Threshold (PFT)
            </div>
            <div style="
                width: 100%; height: 14px; 
                background: linear-gradient(to right, {gradient_css});
                border-radius: 3px; margin-bottom: 6px;
            "></div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #cccccc;">
                <span>{ticks[0]:.1f} GW</span>
                <span>{ticks[1]:.1f} GW</span>
                <span>{ticks[2]:.1f} GW</span>
                <span>{ticks[3]:.1f} GW</span>
                <span>{ticks[4]:.1f} GW</span>
            </div>
            <div style="font-size: 9px; color: #888888; text-align: center; margin-top: 5px;">
                Continuous Scale (Logarithmic Normalization Stretch)
            </div>
        </div>
        """
        return legend_html
    
    def compile_integrated_map(self, geojson_data, pft_df, output_html="weekly_fire_map.html", cmap_name = 'autumn'):
        """
        Builds a spatiotemporal hourly-precision grid of PFT values over land mass 
        with floating legends and static fire vectors.
        """
        m = folium.Map(location=[45, -100], zoom_start=4, tiles="CartoDB dark_matter")
        
        if pft_df.empty:
            print("[-] Warning: PFT DataFrame is empty. Skipping spatiotemporal mesh processing.")
            return m

        extent = config.bounds 
        grid_res = (90, 108)           
        
        lon_edges = np.linspace(extent[0], extent[1], grid_res[1] + 1)
        lat_edges = np.linspace(extent[2], extent[3], grid_res[0] + 1)
        
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
                        feature = {
                            "type": "Feature",
                            "id": str(cell_idx),
                            "geometry": mapping(land_cell),
                            "properties": {}
                        }
                        grid_features.append(feature)
                        cell_centroids.append((cell_poly.centroid.x, cell_poly.centroid.y, cell_idx))
                        cell_idx += 1

        geo_path_collection = {"type": "FeatureCollection", "features": grid_features}

        # Setup interpolation scales
        clean_pft = pft_df[np.isfinite(pft_df['value'])].copy()
        timestamps = sorted(clean_pft['time'].unique())
        
        vmin = max(clean_pft['value'].min(), 1.0)
        vmax = np.percentile(clean_pft['value'], 99)
        if vmin == vmax: vmax += 1.0
        
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)

        style_dict = {str(idx): {} for idx in range(cell_idx)}

        for ts in timestamps:
            ts_group = clean_pft[clean_pft['time'] == ts]
            if len(ts_group) < 4: continue
            ts_dt = pd.to_datetime(ts)
            
            # Explicitly fix localized pandas timezone checking
            if ts_dt.tz is None:
                ts_localized = ts_dt.tz_localize('UTC')
            else:
                ts_localized = ts_dt.tz_convert('UTC')
            
            unix_sec = str(int(ts_localized.timestamp()))
            
            points = ts_group[['lon', 'lat']].values
            values = ts_group['value'].values
            
            grid_coords = [(c[0], c[1]) for c in cell_centroids]
            interpolated_values = griddata(points, values, grid_coords, method='linear')
            
            for val, cell_info in zip(interpolated_values, cell_centroids):
                c_idx = str(cell_info[2])
                if np.isnan(val) or val < vmin:
                    style_dict[c_idx][unix_sec] = {'color': '#000000', 'opacity': 0.0}
                else:
                    rgba = cmap(norm(val))
                    hex_color = mcolors.to_hex(rgba)
                    style_dict[c_idx][unix_sec] = {
                        'color': hex_color,
                        'opacity': 0.45
                    }

        TimeSliderChoropleth(
            data=geo_path_collection,
            styledict=style_dict,
            name="Predictive PFT Forward Mesh Grid",
            stroke_width=0.0,
            date_options="MM:DD:YY:hhmm"
        ).add_to(m)

        legend_html_content = self._generate_html_legend(vmin, vmax, cmap_name=cmap_name)
        m.get_root().html.add_child(folium.Element(legend_html_content))

        if geojson_data and geojson_data.get("features"):
            folium.GeoJson(
                geojson_data,
                name="Active Fires",
                style_function=lambda x: {
                    'fillColor': '#e74c3c',
                    'color': '#c0392b',
                    'weight': 2,
                    'fillOpacity': 0.6
                },
                # Tooltip matches fields to show on hover (minus FRP)
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'fireid', 't_start', 'status'],
                    aliases=['Fire Name:', 'Fire ID:', 'Start Date:', 'Status:'],
                    localize=True
                ),
                # Popup updated to switch 't' for 't_start' and scrub 'meanfrp'
                popup=folium.GeoJsonPopup(
                    fields=['name', 'fireid', 't_start', 'status'],
                    aliases=['Fire Name:', 'Fire ID:', 'Start Date:', 'Status:'],
                    labels=True
                )
            ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(output_html)
        print(f"[+] Spatiotemporal dashboard generated with hourly slider and colorbar: '{output_html}'")
        return m
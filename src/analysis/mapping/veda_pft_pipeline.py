import io
import os
import base64
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import folium
from folium.plugins import TimestampedGeoJson, TimeSliderChoropleth
from shapely.geometry import shape, mapping, MultiPolygon, Polygon
import cartopy.io.shapereader as shapereader
from shapely.ops import unary_union

# from base_pipeline import BaseAtmosphericPipeline
import analysis.mapping.config as config

class VedaPftLandmaskedPipeline:
    def __init__(self):
        self.unified_land_mask = self._build_static_land_mask()

    def _build_static_land_mask(self):
        """Builds a unified land mass for US and Canada while masking out major inland water bodies."""
        # 1. Fetch Sovereign Ground polygons
        country_shp = shapereader.natural_earth(
            resolution='50m', category='cultural', name='admin_0_countries'
        )
        country_reader = shapereader.Reader(country_shp)
        land_geoms = [
            country.geometry for country in country_reader.records() 
            if country.attributes['NAME'] in ['United States of America', 'Canada']
        ]
        unified_ground = unary_union(land_geoms)

        # 2. Fetch Major Lacustrine/Inland Water Body shapes
        lakes_shp = shapereader.natural_earth(
            resolution='50m', category='physical', name='lakes'
        )
        lakes_reader = shapereader.Reader(lakes_shp)
        lake_geoms = [lake.geometry for lake in lakes_reader.records()]
        unified_lakes = unary_union(lake_geoms)

        # 3. Punch holes through the landmass geometry where lakes are located
        clean_landmass = unified_ground.difference(unified_lakes)
        return clean_landmass
    
    def _inject_slider_hour_override(self, folium_map):
        """Injects a custom JavaScript DOM mutation string into the Map output header.

        Forces the TimeSlider widget to display sub-daily timestamps in %Y-%m-%d %H:%M format.
        """
        custom_js = """
        <script>
        $(document).ready(function() {
            function overrideSliderOutput() {
                var label = $(".time-slider-label");
                if (label.length > 0) {
                    // Locate the Leaflet range slider component input instance
                    var sliderInput = $("input.slider");
                    if (sliderInput.length > 0) {
                        sliderInput.on("input change", function() {
                            var val = $(this).val();
                            // Parse out numerical seconds into a valid UTC time string representation
                            var dateObj = new Date(parseInt(val) * 1000);
                            if (!isNaN(dateObj.getTime())) {
                                var yyyy = dateObj.getUTCFullYear();
                                var mm = String(dateObj.getUTCMonth() + 1).padStart(2, '0');
                                var dd = String(dateObj.getUTCDate()).padStart(2, '0');
                                var hh = String(dateObj.getUTCHours()).padStart(2, '0');
                                var min = String(dateObj.getUTCMinutes()).padStart(2, '0');
                                label.text(yyyy + "-" + mm + "-" + dd + " " + hh + ":" + min + " UTC");
                            }
                        });
                        // Trigger an initial execution loop scan pass
                        sliderInput.trigger("change");
                    }
                }
            }
            // Execute loops to handle slight delays in Leaflet control injections
            setTimeout(overrideSliderOutput, 600);
            setTimeout(overrideSliderOutput, 1500);
        });
        </script>
        """
        folium_map.get_root().header.add_child(folium.Element(custom_js))

    def fetch_veda_fires(self, days_back=1, limit_per_region=500):
        """Fetches fire detections for the current day."""
        now = datetime.utcnow()
        start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"{config.VEDA_BASE_URL}/collections/public.eis_fire_lf_perimeter_nrt/items"
        
        combined_features = []
        for name, bbox in config.VEDA_REGIONS.items():
            params = {"limit": limit_per_region, "datetime": f"{start_date}/", "bbox": bbox}
            try:
                response = requests.get(url, params=params, timeout=20)
                if response.status_code == 200:
                    combined_features.extend(response.json().get("features", []))
            except Exception as e:
                print(f"Error fetching region {name}: {e}")
        return {"type": "FeatureCollection", "features": combined_features}

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
        
        # Sample points across the spectrum to construct a smooth gradient
        samples = np.linspace(0, 1, 6)
        hex_colors = [mcolors.to_hex(cmap(s)) for s in samples]
        
        # Calculate sample interval tick values matching log space distribution
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

    def compile_integrated_map(self, geojson_data, pft_df, output_html="weekly_fire_map.html"):
        """
        Builds a spatiotemporal hourly-precision grid of PFT values over land mass 
        with floating legends and static fire vectors.
        """
        m = folium.Map(location=[45, -100], zoom_start=4, tiles="CartoDB dark_matter")
        
        if pft_df.empty:
            print("[-] Warning: PFT DataFrame is empty. Skipping spatiotemporal mesh processing.")
            return m

        # ==========================================================
        # STEP 1: COMPUTE SPATIOTEMPORAL GEOMESH OVER LAND
        # ==========================================================
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
        cmap = plt.get_cmap('autumn')

        style_dict = {str(idx): {} for idx in range(cell_idx)}

        for ts in timestamps:
            ts_group = clean_pft[clean_pft['time'] == ts]
            if len(ts_group) < 4: continue
            
            # CRITICAL FOR HOURLY PRECISION:
            # We explicitly convert the datetime to a Unix epoch in seconds.
            # Folium's underlying time-slider parses integers as seconds, preserving accurate hours/minutes
            unix_sec = str(int(pd.to_datetime(ts).timestamp()))
            
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
                        'opacity': 0.65
                    }

        # 2. Inject TimeSliderChoropleth Layer to the active map canvas
        # Note: By mapping raw unix timestamp fields, the Javascript timeline widget 
        # displays the sub-daily timestamp information securely across forecast loops.
        TimeSliderChoropleth(
            data=geo_path_collection,
            styledict=style_dict,
            name="Predictive PFT Forward Mesh Grid",
            stroke_width=0.0,
            date_options="MM:DD:YY:hh:mm"
        ).add_to(m)

        # ==========================================================
        # STEP 2: INJECT ADAPTIVE CSS COLORBAR LEGEND
        # ==========================================================
        legend_html_content = self._generate_html_legend(vmin, vmax, cmap_name='autumn')
        m.get_root().html.add_child(folium.Element(legend_html_content))

        # self._inject_slider_hour_override(m)

        # ==========================================================
        # STEP 3: OVERLAY TODAY'S STATIC ACTIVE FIRE PERIMETERS
        # ==========================================================
        if geojson_data and geojson_data.get("features"):
            folium.GeoJson(
                geojson_data,
                name="24-Hour Active Fire Footprints",
                style_function=lambda x: {
                    'fillColor': '#e74c3c',
                    'color': '#c0392b',
                    'weight': 2,
                    'fillOpacity': 0.6
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['fireid', 'meanfrp', 'farea'],
                    aliases=['Fire ID:', 'FRP (MW):', 'Area (km²):'],
                    localize=True
                ),
                popup=folium.GeoJsonPopup(
                    fields=['fireid', 't', 'meanfrp', 'farea'],
                    labels=True
                )
            ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(output_html)
        print(f"[+] Spatiotemporal dashboard generated with hourly slider and colorbar: '{output_html}'")
        return m
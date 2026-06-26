import io
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import analysis.mapping.config as config
from analysis.mapping.fire_map_base import FireMapBase
from utils.constants import CLIENTS_DIR, FIRMS_KEY_FNAME

class FirmsPftLandmaskedPipeline(FireMapBase):
    def fetch_fires(self, days_back=1, csv_path="active_fires_summary.csv", cluster_threshold_km=3.0):
        """
        Fetches real-time fire points from FIRMS over configured bounding boxes,
        clusters nearby points, constructs bounding polygons, aggregates attributes,
        and pipes standardized features to the CSV exporter.
        """
        sensor = "VIIRS_NOAA20_NRT"
        firms_key_path = os.path.join(CLIENTS_DIR, FIRMS_KEY_FNAME)
        if os.path.exists(firms_key_path):
            with open(os.path.join(CLIENTS_DIR, FIRMS_KEY_FNAME), 'r') as f:
                api_key = f.read()
        else:
            raise RuntimeError(f"Make sure your FIRMS API key is saved to {firms_key_path}")
        raw_points = []
        now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 1. Harvest raw data chunks from all regions
        for name, bbox in config.VEDA_REGIONS.items():
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{sensor}/{bbox_str}/{days_back}"
            
            try:
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    df_chunk = pd.read_csv(io.StringIO(response.text))
                    raw_points.append(df_chunk)

                    import ipdb; ipdb.set_trace()
            except Exception as e:
                print(f"Error fetching FIRMS data for region {name}: {e}")

        if not raw_points or sum(len(df) for df in raw_points) == 0:
            print("[-] Warning: No real-time FIRMS active points found for the requested criteria.")
            return {"type": "FeatureCollection", "features": []}

        # Combine all harvested points and drop any potential duplicates across bounding box seams
        df_all = pd.concat(raw_points, ignore_index=True).drop_duplicates(subset=['latitude', 'longitude', 'acq_date', 'acq_time'])
        
        # Standardize acquisition time vectors
        df_all['acq_time_str'] = df_all['acq_time'].astype(int).astype(str).str.zfill(4)
        df_all['utc_datetime'] = pd.to_datetime(df_all['acq_date'] + ' ' + df_all['acq_time_str'], format="%Y-%m-%d %H%M")

        # 2. Perform Spatial Hierarchical Clustering
        # Convert kilometer threshold to rough degree equivalents for fast planar evaluation (1 km ≈ 0.009 degrees)
        degree_threshold = cluster_threshold_km * 0.009
        coords = df_all[['longitude', 'latitude']].values
        
        if len(coords) > 1:
            dist_matrix = pdist(coords)
            Z = linkage(dist_matrix, method='complete')
            df_all['cluster_id'] = fcluster(Z, degree_threshold, criterion='distance')
        else:
            df_all['cluster_id'] = 1

        combined_features = []

        # 3. Aggregate clusters into complex spatial geometries and metrics
        for c_id, group in df_all.groupby('cluster_id'):
            cluster_points = group[['longitude', 'latitude']].values
            
            # Identify absolute global temporal bounds per clustered event
            global_min_start = group['utc_datetime'].min().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Cumulative calculations for physical properties
            total_frp = float(group['frp'].mean())
            # Estimate area based on individual sensor pixel footprints (approx 0.375km per sub-pixel scan)
            total_area_km2 = float((group['scan'] * group['track']).sum())
            
            # Generate geometry layout dynamically based on cluster size density
            if len(cluster_points) >= 3:
                try:
                    # Construct a Convex Hull wrapper for dense multi-point arrays
                    hull = ConvexHull(cluster_points)
                    hull_coords = cluster_points[hull.vertices]
                    # Close the linear ring for standard GeoJSON compliance
                    closed_ring = np.vstack([hull_coords, hull_coords[0]])
                    geom_structure = {
                        "type": "Polygon",
                        "coordinates": [closed_ring.tolist()]
                    }
                except Exception:
                    # Fallback to standard Point Centroid if points are co-linear
                    centroid = cluster_points.mean(axis=0)
                    geom_structure = {
                        "type": "Point",
                        "coordinates": [float(centroid[0]), float(centroid[1])]
                    }
            else:
                # Use mean spatial centroid for small arrays
                centroid = cluster_points.mean(axis=0)
                geom_structure = {
                    "type": "Point",
                    "coordinates": [float(centroid[0]), float(centroid[1])]
                }

            # Map compiled cluster metrics into an OGC Feature dict
            feature = {
                "type": "Feature",
                "geometry": geom_structure,
                "properties": {
                    "fireid": f"CLUSTER_FIRE_{group['acq_date'].min().replace('-', '')}_{c_id}",
                    "farea": round(total_area_km2, 3),
                    "meanfrp": round(total_frp, 2),
                    "t": now_str,
                    "t_start": global_min_start,
                    "t_end": now_str
                }
            }
            combined_features.append(feature)

        geojson_res = {"type": "FeatureCollection", "features": combined_features}
        
        # Process and generate the CSV statistical summary using the aggregated metrics
        self.export_fire_statistics_csv(geojson_res, output_csv=csv_path)
        
        return geojson_res
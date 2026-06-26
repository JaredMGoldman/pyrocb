import requests
from datetime import datetime, timedelta

import analysis.mapping.config as config
from analysis.mapping.fire_map_base import FireMapBase

class VedaPftLandmaskedPipeline(FireMapBase):
    def fetch_fires(self, days_back=1, limit_per_region=500, csv_path = "active_fires_summary.csv"):
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
        
        geojson_res = {"type": "FeatureCollection", "features": combined_features}
        
        # Automatically process and generate the CSV statistical manifest 
        self.export_fire_statistics_csv(geojson_res, output_csv=csv_path)
        
        return geojson_res
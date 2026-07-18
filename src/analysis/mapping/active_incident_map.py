import requests
import pandas as pd
from datetime import datetime
from shapely.ops import transform
import json
import os
import pyproj
from functools import partial
from shapely.geometry import shape
from shapely import Point
import shutil

from analysis.mapping.fire_map_base import FireMapBase

def prune_inactive_fires(inactive_fires, active_fire_csv_path):
    shutil.copy(active_fire_csv_path, active_fire_csv_path.replace('.csv', '_superset.csv'))

    active_fire_df = pd.read_csv(active_fire_csv_path)

    new_fire_df = active_fire_df[~active_fire_df.fire_index_id.isin(inactive_fires)].reset_index(drop=True)

    new_fire_df.to_csv(active_fire_csv_path, index=False)
    print("[+] subsetted csv data to active fires")

class ActiveFirePerimeterPipeline(FireMapBase):
    def __init__(self, *args, **kwargs):
        # Authoritative REST Service Endpoints for Live Public Geospatial Perimeters
        self.us_endpoint = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"
        self.ca_endpoint = "https://services.arcgis.com/wjcPoefzjpzCgffS/ArcGIS/rest/services/Active_Wildfire_Perimeters_in_Canada/FeatureServer/0/query"
        self.ca_points_endpoint = "https://services.arcgis.com/wjcPoefzjpzCgffS/arcgis/rest/services/activefires/FeatureServer/0/query"
        super().__init__(*args, **kwargs)

    def _query_arcgis_featureserver(self, url, where_clause="1=1", out_fields="*"):
        """
        Helper function to handle robust paginated GeoJSON extraction from 
        ArcGIS REST endpoints to bypass MaxRecordCount caps safely.
        """
        combined_features = []
        offset = 0
        limit = 2000  # Standard ArcGIS default chunk limits
        
        while True:
            params = {
                "where": where_clause,
                "outFields": out_fields,
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": limit,
                "outSR": "4326"  # Guarantee WGS84 coordinates out-of-the-box
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    print(f"[-] HTTP Error querying FeatureServer layer: {response.status_code}")
                    break
                    
                data = response.json()
                features = data.get("features", [])
                if not features:
                    break
                    
                combined_features.extend(features)
                if len(features) < limit:
                    break  # Reached the final page
                offset += len(features)
            except Exception as e:
                print(f"[-] Critical failure during network retrieval chunk: {e}")
                break
                
        return combined_features

    def _process_us_perimeters(self):
        """
        Helper function to extract, normalize, and structurally clean 
        active US Interagency perimeters using the WFIGS schema mapping.
        """
        print("[*] Accessing US WFIGS Current Interagency Fire Perimeters...")
        # WFIGS filters out uncertified inactive historical records natively inside 'current'
        raw_features = self._query_arcgis_featureserver(self.us_endpoint)
        standardized = []
        
        for feature in raw_features:
            props = feature.get("properties", {})
            geom = feature.get("geometry", None)
            if not geom or geom.get("type") not in ["Polygon", "MultiPolygon"]:
                continue
                
            # Cross-reference attributes safely from WFIGS schema specs
            # Unique IRWIN ID string identifier (e.g. {ABCD-1234...})
            irwin_id = props.get("attr_IrwinID", props.get("poly_IRWINID", "UNKNOWN_US_ID")).replace('{','').replace('}','')
            
            # Fire Status evaluation rules matching NWCG standards
            is_out = props.get("attr_FireOutDateTime", None) is not None
            status = "Inactive/Out" if is_out else "Active"
            
            # Handle timestamps (ArcGIS represents time in Epoch milliseconds)
            start_ms = props.get("attr_FireDiscoveryDateTime", props.get("poly_CreateDate", None))
            start_date_iso = None
            if start_ms:
                try:
                    start_date_iso = datetime.utcfromtimestamp(start_ms / 1000.0).strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass
            
            # Area normalization (convert GIS Acres standard to Square Kilometers)
            gis_acres = props.get("poly_GISAcres", props.get("attr_IncidentSize", 0.0))
            area_km2 = float(gis_acres) * 0.00404686 if gis_acres else 0.0
            
            standardized.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "fireid": irwin_id,
                    "name": props.get("poly_IncidentName", "Unnamed US Incident"),
                    "farea": round(area_km2, 3),
                    "t_start": start_date_iso,
                    "t_end": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status": status,
                    "country": "USA"
                }
            })
        print(f"[+] Successfully structured {len(standardized)} US active fire tracks.")
        return standardized
    
    def _process_canadian_perimeters(self):
        """
        Extracts Canadian perimeters (polygons) and pools all corresponding 
        attribute points to reconcile metrics using the earliest start times 
        and the most severe operational fire status. Area is derived directly 
        from the polygon geometry.
        """
        print("[*] Harvesting Canadian Polygon Perimeters...")
        raw_polygons = self._query_arcgis_featureserver(self.ca_endpoint)
        
        print("[*] Harvesting Canadian Live Incident Information Points...")
        raw_points = self._query_arcgis_featureserver(self.ca_points_endpoint)

        if not raw_polygons:
            print("[-] Warning: No Canadian perimeters found to assign features against.")
            return []

        # 1. Parse and initialize structured perimeter objects
        polygon_inventory = []
        for poly_feat in raw_polygons:
            geom = poly_feat.get("geometry", None)
            if not geom or geom.get("type") not in ["Polygon", "MultiPolygon"]:
                continue
            
            try:
                shapely_poly = shape(geom)
                props = poly_feat.get("properties", {})
                polygon_inventory.append({
                    "shapely_obj": shapely_poly,
                    "raw_geometry": geom,
                    "fallback_id": next((props[k] for k in ['FIRE_ID', 'fire_id', 'FIREID'] if k in props), "UNKNOWN_CA_ID"),
                    "pooled_points": [] # Container to catch all overlapping incident points
                })
            except Exception:
                continue

        # 2. Map every incoming point to its corresponding polygon pool
        print(f"[*] Pooling {len(raw_points)} points into {len(polygon_inventory)} perimeter bins...")
        for pt_feat in raw_points:
            pt_geom = pt_feat.get("geometry", None)
            if not pt_geom or pt_geom.get("type") != "Point":
                continue

            try:
                pt_coords = pt_geom.get("coordinates")
                shapely_point = Point(pt_coords[0], pt_coords[1])
                pt_props = pt_feat.get("properties", {})

                best_match_idx = None
                closest_distance = float('inf')

                for idx, poly_item in enumerate(polygon_inventory):
                    if poly_item["shapely_obj"].contains(shapely_point):
                        best_match_idx = idx
                        break
                    
                    dist = shapely_point.distance(poly_item["shapely_obj"])
                    if dist < closest_distance:
                        closest_distance = dist
                        best_match_idx = idx

                if best_match_idx is not None:
                    polygon_inventory[best_match_idx]["pooled_points"].append(pt_props)

            except Exception as ex:
                print(f"[-] Error routing point element to spatial pool: {ex}")
                continue

        # Helper lambda to project to an equal-area projection (Canada Albers) to get accurate km²
        # This automatically handles WGS84 Lat/Lon geometries seamlessly
        project_to_km2 = partial(
            pyproj.transform,
            pyproj.Proj("epsg:4326"), # WGS84
            pyproj.Proj("epsg:3978")  # Canada Atlas Lambert (Meters)
        )

        # 3. Aggregate and resolve each pool bucket down to a single standardized feature
        standardized = []
        status_priority = ["OUT", "EXTINGUISHED", "UC", "UNDER CONTROL", "BH", "BEING HELD", "OC", "OUT OF CONTROL"]

        for idx, poly_item in enumerate(polygon_inventory):
            pooled_pts = poly_item["pooled_points"]
            shapely_poly = poly_item["shapely_obj"]

            # --- GEOMETRIC AREA CALCULATION ---
            # Projects to Canada Albers (meters), calculates m², then converts to km²
            try:
                projected_poly = transform(project_to_km2, shapely_poly)
                area_km2 = projected_poly.area / 1_000_000.0
            except Exception:
                # Fallback scaling if pyproj/transform fails (rough degree to km conversion baseline)
                # 1 sq degree near Canada (~50°N) is roughly 7200 sq km, but projection is vastly superior
                area_km2 = shapely_poly.area * 7200.0 
            
            # Case A: Polygon has one or more matching information points
            if pooled_pts:
                fire_id = next((p[k] for p in pooled_pts for k in ['fire_id', 'id', 'FIREID', 'FIRE_ID', 'ObjectId'] if k in p), poly_item["fallback_id"])
                incident_name = next((p[k] for p in pooled_pts for k in ['name', 'fire_name', 'FIRE_NAME', 'Fire_Name'] if k in p), f"Fire {fire_id}")
                
                # RECONCILIATION RULE 1: Earliest Start Date Extraction
                earliest_start = None
                for p in pooled_pts:
                    start_val = next((p[k] for k in ['start_date', 'START_DATE', 'discovered', 'Start_Date'] if k in p), None)
                    if start_val:
                        try:
                            dt_val = pd.to_datetime(float(start_val), unit='ms' if float(start_val) > 2e9 else 's', utc=True)
                            if earliest_start is None or dt_val < earliest_start:
                                earliest_start = dt_val
                        except Exception:
                            pass
                
                start_date_iso = earliest_start.strftime("%Y-%m-%dT%H:%M:%SZ") if earliest_start else None

                # RECONCILIATION RULE 2: Most Extreme Status Evaluation
                highest_severity_rank = -1
                chosen_raw_status = "Active"
                
                for p in pooled_pts:
                    raw_status_str = str(next((p[k] for k in ['status', 'STATUS', 'stage_of_c', 'Stage_of_Control'] if k in p), "Active")).upper().strip()
                    rank = next((i for i, s in enumerate(status_priority) if s in raw_status_str), 2)
                    if rank > highest_severity_rank:
                        highest_severity_rank = rank
                        chosen_raw_status = raw_status_str

                status = "Inactive/Out" if highest_severity_rank <= 1 else status_priority[highest_severity_rank]
                
                # Append compiled data (using newly calculated geometric area)
                standardized.append({
                    "type": "Feature",
                    "geometry": poly_item["raw_geometry"],
                    "properties": {
                        "fireid": f"CA_{fire_id}",
                        "name": f"{incident_name}",
                        "farea": round(area_km2, 3),
                        "t_start": start_date_iso,
                        "t_end": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "status": status,
                        "raw_status": chosen_raw_status,
                        "country": "CAN"
                    }
                })

            # Case B: Catch-all fallback if polygon track exists without point tracking data
            else:
                standardized.append({
                    "type": "Feature",
                    "geometry": poly_item["raw_geometry"],
                    "properties": {
                        "fireid": f"CA_{poly_item['fallback_id']}",
                        "name": f"{poly_item['fallback_id']}",
                        "farea": round(area_km2, 3),
                        "t_start": None,
                        "t_end": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "status": "Active",
                        "country": "CAN"
                    }
                })
        standardized = [item for item in standardized if item['properties']['name'] != "UNKNOWN_CA_ID"]
        print(f"[+] Successfully structured and resolved {len(standardized)} pooled Canadian tracks.")
        return standardized

    def export_fire_statistics_csv(self, geojson_data, output_csv="active_fires_summary.csv"):
        """
        Parses compiled fire feature clusters to compile structural statistics,
        converting vector boundaries to Well-Known Text (WKT) geometries.
        """
        if not geojson_data or not geojson_data.get("features"):
            print("[-] Warning: No active fire features available to parse into CSV summary.")
            return None

        records = []
        for idx, feature in enumerate(geojson_data.get("features", [])):
            # --- TWO-LINE INDEX INJECTION FIX ---
            fire_index_key = f"fire_idx_{idx}"
            feature["properties"]["fire_index_id"] = fire_index_key
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
                
                # Append metrics plus new operational control records
                records.append({
                    "fire_id": fire_id,
                    "fire_index_id": fire_index_key,
                    "name": props.get("name", "UNKNOWN"),
                    "country": props.get("country", "UNKNOWN"),
                    "status": props.get("status", "Active"),
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "duration_days": duration_days,
                    "peak_area_km2": peak_area,
                    "wkt_geometry": wkt_geometry
                })
            except Exception as ex:
                print(f"[-] Bypassed single feature profile calculation: {ex}")
                continue
                
        if records:
            df = pd.DataFrame(records)
            df = df.sort_values(by="peak_area_km2", ascending=False)
            df = df[df['peak_area_km2'] > 10].reset_index(drop=True)
            df = df.drop(df[df.name == "UNKNOWN_CA_ID"].index).reset_index(drop=True)
            df.to_csv(output_csv, index=False)
            print(f"[+] Active fire statistical inventory saved successfully: '{output_csv}'")
            return df
        return None

    def fetch_fires(self, csv_path="active_fires_summary.csv", only_active=False):
        """
        Primary execution entry point. Combines cleaned US and Canadian vector layers,
        filters by operational status if requested, and saves the global statistical inventory.
        """
        # Execute helper data processors independently for clean troubleshooting flows
        us_features = self._process_us_perimeters()
        ca_features = self._process_canadian_perimeters()
        
        all_features = us_features + ca_features
        
        # Apply strict operational status filtration rules if requested
        if only_active:
            all_features = [f for f in all_features if f["properties"]["status"] == "Active"]
            print(f"[*] Filtered data down to {len(all_features)} strictly Active perimeters.")

        geojson_res = {
            "type": "FeatureCollection",
            "features": all_features
        }
        
        # Export processed results to the final CSV data model layout
        self.export_fire_statistics_csv(geojson_res, output_csv=csv_path)

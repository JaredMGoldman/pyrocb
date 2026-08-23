import requests
import pandas as pd
from datetime import datetime
from shapely.ops import transform
from shapely import wkt
import pyproj
from functools import partial
from shapely.geometry import shape, box, mapping, Point
from shapely import intersects
import shutil

from analysis.mapping.fire_map_base import FireMapBase

def prune_inactive_fires(fire_info_csv, rave_csv, bbox):
    shutil.copy(fire_info_csv, fire_info_csv.replace('.csv', '_superset.csv'))
    shutil.copy(rave_csv, rave_csv.replace('.csv', '_superset.csv'))

    fire_info_df = pd.read_csv(fire_info_csv)
    rave_data_df = pd.read_csv(rave_csv)

    zero_frp_mask = rave_data_df.groupby('fire_index_id')['total_rave_frp'].sum() == 0
    invalid_fires = zero_frp_mask[zero_frp_mask].index.tolist()

    # 1. Parse WKT strings row-by-row
    fire_info_df['geometry'] = fire_info_df['wkt_geometry'].apply(wkt.loads)

    # 2A. Using Shapely 2.0+ Vectorized API (Fastest pure-shapely method)
    is_contained = intersects(bbox, fire_info_df['geometry'].values)

    unbounded = fire_info_df[~is_contained].fire_index_id.tolist() 
    invalid = list(set(invalid_fires) | set(unbounded))

    new_rave_df = rave_data_df[~rave_data_df.fire_index_id.isin(invalid)].reset_index(drop=True)
    new_info_df = fire_info_df[~fire_info_df.fire_index_id.isin(invalid)].reset_index(drop=True)

    new_info_df.to_csv(fire_info_csv, index=False)
    new_rave_df.to_csv(rave_csv, index=False)
    print("[+] subsetted csv data to active fires")

class ActiveFirePerimeterPipeline(FireMapBase):
    def __init__(self, *args, **kwargs):
        # Authoritative WFIGS REST Service Endpoints
        self.us_endpoint = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"
        self.us_locations_endpoint = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0/query"
        
        # Esri USA Wildfires v1 Feed (Powers Dashboard & Hawk Fire Tracking)
        self.us_wildfires_v1_endpoint = "https://services9.arcgis.com/RHVPKKiFTONKtxq3/ArcGIS/rest/services/USA_Wildfires_v1/FeatureServer/0/query"
        
        self.ca_endpoint = "https://services.arcgis.com/wjcPoefzjpzCgffS/ArcGIS/rest/services/Active_Wildfire_Perimeters_in_Canada/FeatureServer/0/query"
        self.ca_points_endpoint = "https://services.arcgis.com/wjcPoefzjpzCgffS/arcgis/rest/services/activefires/FeatureServer/0/query"
        super().__init__(*args, **kwargs)

    def _clean_id(self, val):
        if not val or not isinstance(val, str):
            return ""
        return val.replace('{', '').replace('}', '').strip().upper()

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
        Extracts US perimeters across WFIGS and USA_Wildfires_v1 endpoints.
        Deduplicates strictly by normalized IRWIN ID, or by matching Name + Centroid Proximity.
        """
        print("[*] Accessing US WFIGS Current Interagency Fire Perimeters...")
        raw_polygons = self._query_arcgis_featureserver(self.us_endpoint)
        
        print("[*] Accessing US WFIGS Current Interagency Incident Locations...")
        raw_locations = self._query_arcgis_featureserver(self.us_locations_endpoint)

        print("[*] Accessing USA Wildfires v1 Feed (Dashboard Aggregator)...")
        raw_usa_v1 = self._query_arcgis_featureserver(self.us_wildfires_v1_endpoint)

        polygon_map = {}
        
        # 1. Index perimeter polygons by normalized IrwinID and Name
        for feature in raw_polygons:
            props = feature.get("properties", {})
            geom = feature.get("geometry", None)
            if not geom or geom.get("type") not in ["Polygon", "MultiPolygon"]:
                continue

            irwin_id = self._clean_id(props.get("poly_IRWINID", props.get("attr_IrwinID", "")))
            name = props.get("poly_IncidentName", props.get("attr_IncidentName", "")).strip().upper()
            
            if irwin_id:
                polygon_map[irwin_id] = feature
            if name and name not in polygon_map:
                polygon_map[name] = feature

        all_incidents = raw_locations + raw_usa_v1
        standardized = []
        processed_irwin_ids = set()
        seen_entries = [] # List of tuples: (name_upper, shapely_centroid)

        # 2. Process all point/incident records
        for loc_feat in all_incidents:
            props = loc_feat.get("properties", {})
            irwin_id = self._clean_id(props.get("IrwinID", props.get("attr_IrwinID", props.get("poly_IRWINID", ""))))
            name = props.get("IncidentName", props.get("attr_IncidentName", props.get("poly_IncidentName", "Unnamed US Incident"))).strip()
            name_key = name.upper()

            # Deduplication Check 1: Strict IRWIN ID match
            if irwin_id and irwin_id in processed_irwin_ids:
                continue

            final_geom = None
            poly_acres = None

            # Preference A: Use matching perimeter polygon
            match_feat = polygon_map.get(irwin_id) or polygon_map.get(name_key)
            if match_feat:
                final_geom = match_feat.get("geometry")
                poly_acres = match_feat.get("properties", {}).get("poly_GISAcres", None)

            # Preference B: Fallback to 0.5-degree bounding box around point location
            if not final_geom:
                geom = loc_feat.get("geometry", None)
                lon, lat = None, None

                if geom and geom.get("type") == "Point":
                    coords = geom.get("coordinates", [])
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                
                if lon is None or lat is None:
                    lon = props.get("attr_InitialLongitude", props.get("Longitude", props.get("longitude", None)))
                    lat = props.get("attr_InitialLatitude", props.get("Latitude", props.get("latitude", None)))

                if lon is not None and lat is not None:
                    try:
                        lon, lat = float(lon), float(lat)
                        bbox_polygon = box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25)
                        final_geom = mapping(bbox_polygon)
                    except (ValueError, TypeError):
                        final_geom = None

            if not final_geom:
                continue

            # Deduplication Check 2: Same Name AND Close Centroid Distance (< ~10 km / 0.1 deg)
            try:
                shapely_obj = shape(final_geom)
                centroid = shapely_obj.centroid
                
                is_duplicate = False
                for prev_name, prev_centroid in seen_entries:
                    if prev_name == name_key and centroid.distance(prev_centroid) < 0.1:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue

                seen_entries.append((name_key, centroid))
            except Exception:
                pass

            is_out = props.get("attr_FireOutDateTime", props.get("FireOutDateTime", None)) is not None
            status = "Inactive/Out" if is_out else "Active"

            start_ms = props.get("attr_FireDiscoveryDateTime", props.get("FireDiscoveryDateTime", props.get("poly_CreateDate", None)))
            start_date_iso = None
            if start_ms:
                try:
                    start_date_iso = datetime.utcfromtimestamp(start_ms / 1000.0).strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass

            gis_acres = poly_acres or props.get("DailyAcres", props.get("attr_IncidentSize", props.get("poly_GISAcres", 0.0)))
            area_km2 = float(gis_acres) * 0.00404686 if gis_acres else 0.0

            standardized.append({
                "type": "Feature",
                "geometry": final_geom,
                "properties": {
                    "fireid": irwin_id if irwin_id else f"US_{name_key.replace(' ', '_')}",
                    "name": name,
                    "farea": round(area_km2, 3),
                    "t_start": start_date_iso,
                    "t_end": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status": status,
                    "country": "USA"
                }
            })

            if irwin_id:
                processed_irwin_ids.add(irwin_id)

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
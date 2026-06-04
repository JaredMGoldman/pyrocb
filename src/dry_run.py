from fire_comparison import find_cp 
from shapely import Point


if __name__ == "__main__":
    date1 = "2024-09-07"
    dry_run_fires = [
    (date1, 43.6, -120.9),
    (date1, 43.5, -121),
    (date1, 44.4, -119.6),
    (date1, 46.1, -115.1),
    (date1, 45.6, -114.9),
    (date1, 43.7, -110.3),
    (date1, 34.1, -117.2)]

    cp_set = set()
    for date, lat, lon in dry_run_fires:
        cps = find_cp(date,Point(lon,lat)) 
        cp_set = cp_set | set(cps)
    print(cp_set)
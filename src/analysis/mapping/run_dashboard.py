import os
from os.path import join as os_join
from os.path import exists as os_exists
import shutil

from analysis.mapping.pft_gen_parallel import pull_data, group_client_dses, \
                                        calc_pfts, calc_soundings, \
                                        parse_to_dataframe
import analysis.mapping.config as config
from analysis.mapping.active_incident_map import prune_inactive_fires
from analysis.mapping.copy_util import upload_simplified
from analysis.mapping.signal_cache import SignalCache
from analysis.mapping.rave_oper_client import RAVEOperClient
from analysis.mapping.canadian_frp_prediction import execute_predictive_fire_pipeline

from utils.io_utils import safe_copy

def _copy_current(fname):
    if not config.DEBUG_MODE:
        safe_copy(os_join(config.today_dir, fname), 
              os_join(config.current_dir, fname))

def FETCH():
    print('fetching active fire polygons...')
    config.active_fire_class()\
            .fetch_fires(csv_path = os_join(config.today_dir, 
                                            config.active_fire_fname))
    _copy_current(config.active_fire_fname)

def RAVE():
    print('pulling rave values for active fires...')
    pipeline = RAVEOperClient(csv_path = os_join(config.today_dir, config.active_fire_fname),
                   download_dir = config.rave_cache,
                   output_csv = os_join(config.today_dir, config.active_rave_fn))
    downloaded_files = pipeline.download_rave_files(last_n_days=config.rave_lookback, 
                                                    reference_date=config.now_dt)
    dropped_fires = pipeline.extract_frp_data_parallel_files(downloaded_files, config.max_workers)
    prune_inactive_fires(dropped_fires,
                         os_join(config.today_dir, 
                                 config.active_fire_fname))
                        
    _copy_current(config.active_rave_fn)

def FRP_CAN():
    print('running canadian frp prediction...')
    frp_csv = execute_predictive_fire_pipeline( target_dt = config.now_dt, 
                                                out_dir = config.today_dir)
    _copy_current(config.can_frp_fname)
    return frp_csv

def DB():
    print('generating pft sounding database...')
    dses = pull_data(config.now_date, config.lats, config.lons, 
                     config.fxx_range, config.clients, 
                     config.max_workers, config.fxx_freq)
    client_dses = group_client_dses(dses, config.fx_names)
    cache = SignalCache(db_path=os_join(config.today_dir, config.snd_cache_fn))
    calc_soundings(client_dses, config.max_workers, cache)
    return cache

def PFT(cache):
    print('calculating active pfts...')
    pfts = calc_pfts(cache, config.max_workers)
    df_pfts_calculated = parse_to_dataframe(pfts)
    df_pfts_calculated.to_csv(os_join(config.today_dir, config.pft_fname), index = False)
    _copy_current(config.pft_fname)

def MAP():
    print('generating html file...')
    config.active_fire_class().compile_integrated_map(
        manifest_path= os_join(config.today_dir, config.active_fire_fname), 
        pft_path = os_join(config.today_dir, config.pft_fname),
        output_html = os_join(config.today_dir, config.html_fname),
        cmap_name = config.cmap_name,
        frp_csv_path = os_join(config.today_dir, config.can_frp_fname)
    )
    _copy_current(config.html_fname)

    upload_simplified(  os_join(config.today_dir, config.html_fname), 
                        os_join(config.REMOTE_DIR, config.html_fname),
                        hostname = config.HOSTNAME, username=config.USERNAME)
    if not config.DEBUG_MODE:
        upload_simplified(os_join(config.today_dir, config.html_fname), 
                          os_join(config.REMOTE_DIR, "latest.html"),
                          hostname = config.HOSTNAME, username=config.USERNAME)

def run_pipeline():
    if not os_exists(os_join(config.today_dir, config.active_fire_fname)):
        print('missing active fire polygons.')
        FETCH()
        RAVE()
        FRP_CAN()
    if not os_exists(os_join(config.today_dir, config.active_rave_fn)):
        print('missing active fire rave values.')
        RAVE()
        FRP_CAN()
    if not os_exists(os_join(config.today_dir, config.can_frp_fname)):
        print('missing active fire frp predictions.')
        FRP_CAN()
    if not os_exists(os_join(config.today_dir, config.pft_fname)):
        print('missing pft mesh.')
        cache = DB()
        PFT(cache)
    MAP()

if __name__ == "__main__":
    run_pipeline()
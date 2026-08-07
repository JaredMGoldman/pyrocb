from data.clients.esi_client import ESIClient
from data.clients.firms_client import FirmsClient
from data.clients.rave_client import RAVEClient
from data.clients.rrfs_client import RRFSClient
from data.clients.modis_client import MODISClient

hrrr_headers = [f'hrrr_{feat}' for feat in ['dpt', 'u', 'v', 't', 'rh', 'tp']] #, 'mstav', 'sdwe']]

client_query_specs = [
    {"name": "esi", "client_ctor": ESIClient, "client_kwargs": {}, "vars": ["DFPPM"]},
    {"name": "can_hrrr", "client_ctor": RRFSClient, "client_kwargs": {}, "vars": [
        ":tp:",
        ":r:1000",
        ":10u:",
        ":10v:",
        ":2t:",
        ":2d:",
    ]},
    # {"name": "modis", "client_ctor": MODISClient, "client_kwargs": {}, "vars": ["MaxFRP"]},
    {"name": "rave", "client_ctor": RAVEClient, "client_kwargs": {}, "vars": ["FRP_MEAN", "FRP_SD"]},
]
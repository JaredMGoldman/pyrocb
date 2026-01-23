import pandas as pd
import numpy as np
import datetime

from timezonefinder import TimezoneFinder
import pytz

def structures_timeseries(df, sit209_data):
    #get the fire incident number, lat, and lon
    incident_number = df['irwinID'].unique()[0]
    fire_lat = df['Lat Fire'].unique()[0]
    fire_lon = df['Lon Fire'].unique()[0]
    
    sit209_data_fire = sit209_data[sit209_data['IRWIN_IDENTIFIER']==incident_number]
    if len(sit209_data_fire)==0:
        return 1
    
    #do the time zone conversion
    obj=TimezoneFinder() #initialize the timezone finder
    tz = obj.timezone_at(lng=fire_lon, lat=fire_lat) #get the timezone
    local = pytz.timezone(tz)
    utc = pytz.utc
    
    #put the start and end times in local time
    loc_dt_start = [local.localize(datetime.datetime.strptime(date, '%m/%d/%Y %H:%M:%S')) for date in sit209_data_fire['REPORT_FROM_DATE'].values]
    loc_dt_end = [local.localize(datetime.datetime.strptime(date, '%m/%d/%Y %H:%M:%S')) for date in sit209_data_fire['REPORT_TO_DATE'].values]
    
    #put them in UTC time
    utc_dt_start = [time_start.astimezone(utc) for time_start in loc_dt_start]
    utc_dt_end = [time_end.astimezone(utc) for time_end in loc_dt_end]
    
    #reassign to UTC time, this DOES keep track of daylight savings (eg +7 is used for PDT, +8 is used for PST)
    sit209_data_fire['Report Start UTC'] = pd.to_datetime(utc_dt_start)
    sit209_data_fire['Report End UTC'] = pd.to_datetime(utc_dt_end)
    sit209_data_fire['Timezone']= tz
    
    # localise the index
    sit209_data_fire = sit209_data_fire.set_index(['Report Start UTC']).tz_localize(None)
        
    ## do the 12z-12z day grouping, based on the UTC times
    start_day_utc=str(df['12Z Start Day'][0])
    start_datetime_utc = np.datetime64(start_day_utc[0:10]+'T12:00')

    structures = sit209_data_fire[['QTY_DESTROYED', 'QTY_DAMAGED', 'QTY_THREATENED_72']].resample('24H',origin=start_datetime_utc).sum().reset_index()

    df_sit209 = pd.concat([structures],axis=1)
    df_sit209.columns=['day','structures_destroyed','structures_damaged','structures_threatened_72']

    df_sit209['day'] = pd.to_datetime(df_sit209['day'].values).strftime('%Y-%m-%d')
    df_sit209=df_sit209.fillna(value=0)
    
    return df_sit209

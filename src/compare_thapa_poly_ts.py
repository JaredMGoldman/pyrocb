# %%
import pandas as pd
#pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=100000)
import warnings


warnings.filterwarnings('ignore')
import seaborn as sns

# %%
#plots two lines, on one or on the same axis
# IN: datax, x axis vector (time)
# IN: datay, 2*n vector of the data
# IN: linelabels, string vector of the labels of each line, to go in the legend
# IN: axis labels, labels of the axis
# IN: title, string for title of figire
# IN: twoaxis, if 1 plot data on two separate axes, else plot on the same axis
# OUT: a graph
def plot_twoline(datax, datay, title, linelabels, axislabels, twoaxis):
    fig, ax = plt.subplots(figsize=(20,6)) #set up the figure
    
    if twoaxis==1:
        ax.plot(datax, datay[0,:], 'ro-') # plot the first line
        ax.set_ylabel(axislabels[0], fontsize=24, color='red') 
        ax2=ax.twinx()
        ax2.plot(datax, datay[1,:], 'bo-') #plot the second line
        ax2.set_ylabel(axislabels[1], fontsize=24, color='blue')
        
    else:
        ax.plot(datax, datay[0,:], 'ro-') # plot the first line
        ax.plot(datax, datay[1,:], 'bo-') # plot the second line
        ax.set_ylabel(axislabels, fontsize=24) 
    #print(np.arange(1,len(datax.values),6))
    plt.xticks(np.arange(1,len(datax.values),6))    
    plt.legend(linelabels,fontsize=24)
    plt.title(title,fontsize=24)
    plt.grid()
    plt.show()

def plot_correlations(df_pws, df_merra_lagged, df_merra_movingaverage,df_rave, fig_title):
    #concatenate all the dataframes
    df_all = pd.concat([df_pws.drop(columns=['Unnamed: 0']),\
                    #df_merra_lagged.drop(columns=['Unnamed: 0', 'frp', 'area', 'num points']),\
                    df_merra_movingaverage.drop(columns=['Unnamed: 0','temp','vpd','wind', 'frp','area', 'num points']),\
                    df_rave['CO']], axis=1)

    corrMatrix = df_all.corr()
    fig,ax=plt.subplots(figsize=(18,18))
    sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, cmap='seismic', annot_kws={'fontsize':16})
    plt.title(fig_title, fontsize=20)
    plt.show()

# %%
# Correlation plots for all days, all fires
rave_all = pd.DataFrame()
pws_all = pd.DataFrame()
merra_lagged_all = pd.DataFrame()
merra_movingaverage_all = pd.DataFrame()

fire_incidents = ['AUGUST COMPLEX', 'BOBCAT', 'DOLAN', 'HOLIDAY FARM','CREEK', 'LAKE', 'CAMERON PEAK', 'PINE GULCH']
for ii in range(len(fire_incidents)):
    rave = pd.read_csv(fire_incidents[ii]+'_Daily_RAVE.csv')
    inds_active = np.where(rave['Mean_FRP']>10)[0] #this is the threshold I used at NOAA
    
    rave=rave.iloc[inds_active]
    pws = pd.read_csv(fire_incidents[ii]+'_Daily_PWS.csv').iloc[inds_active]   
    merra_lagged = pd.read_csv(fire_incidents[ii]+'_Daily_MERRA_Lagged.csv').iloc[inds_active]
    merra_movingaverage = pd.read_csv(fire_incidents[ii]+'_Daily_MERRA_Moving_Average_2.csv').iloc[inds_active]
    plot_correlations(pws, merra_lagged, merra_movingaverage,rave, fire_incidents[ii])
    plot_twoline(merra_lagged['day'], np.stack([pws['pws'], rave['CO']]),fire_incidents[ii] ,\
             [], ['PWS', 'CO (kg)'], 1)
    plot_twoline(merra_lagged['day'], np.stack([merra_movingaverage['vpd'], rave['CO']]),fire_incidents[ii] ,\
             [], ['VPD (hpa)', 'CO (kg)'], 1)
    
    rave_all = pd.concat([rave_all,rave], axis=0)
    pws_all = pd.concat([pws_all,pws], axis=0)
    merra_lagged_all = pd.concat([merra_lagged_all,merra_lagged], axis=0)
    merra_movingaverage_all = pd.concat([merra_movingaverage_all,merra_movingaverage], axis=0)

# %%
inds_high_pws = np.where(pws_all['pws']>1.5)[0]
inds_med_pws = np.where((pws_all['pws']<=1.5)&(pws_all['pws']>=1))[0]
inds_low_pws = np.where(pws_all['pws']<1)[0]

print(inds_high_pws)
print(inds_med_pws)
print(inds_low_pws)

plot_correlations(pws_all, merra_lagged_all, merra_movingaverage_all,rave_all, 'All 8 Cases')

plot_correlations(pws_all.iloc[inds_high_pws], merra_lagged_all.iloc[inds_high_pws],
                  merra_movingaverage_all.iloc[inds_high_pws],rave_all.iloc[inds_high_pws], 'All 8 Cases High PWS')

plot_correlations(pws_all.iloc[inds_med_pws], merra_lagged_all.iloc[inds_med_pws],
                  merra_movingaverage_all.iloc[inds_med_pws],rave_all.iloc[inds_med_pws], 'All 8 Cases Medium PWS')

plot_correlations(pws_all.iloc[inds_low_pws], merra_lagged_all.iloc[inds_low_pws],
                  merra_movingaverage_all.iloc[inds_low_pws],rave_all.iloc[inds_low_pws], 'All 8 Cases Low PWS')

# %% [markdown]
# ## Load in the time series for August Complex

# %%
df_merra = pd.read_csv('AC_Daily_MERRA.csv').iloc[0:50]
df_rave = pd.read_csv('AC_Daily_RAVE.csv').iloc[0:50]
df_ncar = pd.read_csv('AC_Daily_NCAR_Moisture.csv').iloc[0:50]
df_personnel = pd.read_csv('AC_Daily_Personnel.csv').iloc[0:50]
df_pws = pd.read_csv('AC_Daily_PWS.csv').iloc[0:50]
df_movingaverage = pd.read_csv('AC_Daily_MERRA_movingaverage.csv').iloc[0:50]


# %%
plot_twoline(df_merra['day'], np.stack([df_merra['temp'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max Temperature (K)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['vpd'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max VPD (hPa)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['wind'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max Wind (m/s)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_merra['hd0w0'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max HD0W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd1w0'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max HD1W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd2w0'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max HD2W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd3w0'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max HD3W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd4w0'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max HD4W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd5w0'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Max HD5W0', 'CO (kg)'], 1)

plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd0w0_MA'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['HD0W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd1w0_MA'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['HD1W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd2w0_MA'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['HD2W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd3w0_MA'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['HD3W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd4w0_MA'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['HD4W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd5w0_MA'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['HD5W0 Moving', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_ncar['mlive']*100, df_rave['CO']]), 'August Complex Polygons',\
             [], ['Live FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_ncar['mdead']*100, df_rave['CO']]), 'August Complex Polygons',\
             [], ['Dead FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_personnel['personnel'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['Personnel (#)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_pws['pws'], df_rave['CO']]), 'August Complex Polygons',\
             [], ['PWS', 'CO (kg)'], 1)

# %%
#concatenate all the dataframes
df_all = pd.concat([df_merra.drop(columns=['Unnamed: 0', 'frp', 'area', 'num points']),\
                    df_movingaverage.drop(columns=['Unnamed: 0','temp','vpd','wind', 'frp', 'area', 'num points']),\
                    #df_ncar.drop(columns=['Unnamed: 0']),\
                    #df_personnel.drop(columns=['Unnamed: 0']),\
                    df_pws.drop(columns=['Unnamed: 0']),\
                    df_rave['CO']], axis=1)
df_all
corrMatrix = df_all.corr()
fig,ax=plt.subplots(figsize=(18,18))
sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, cmap='seismic')
plt.title('August Complex Active Days')
plt.show()

# %% [markdown]
# ## Bobcat Fire

# %%
df_merra = pd.read_csv('BOBCAT_Daily_MERRA.csv').iloc[0:18]
df_rave = pd.read_csv('BOBCAT_Daily_RAVE.csv').iloc[0:18]
df_ncar = pd.read_csv('BOBCAT_Daily_NCAR_Moisture.csv').iloc[0:18]
df_personnel = pd.read_csv('BOBCAT_Daily_Personnel.csv').iloc[0:18]
df_pws = pd.read_csv('BOBCAT_Daily_PWS.csv').iloc[0:18]
df_movingaverage = pd.read_csv('BOBCAT_Daily_MERRA_movingaverage.csv').iloc[0:18]


# %%
plot_twoline(df_merra['day'], np.stack([df_merra['temp'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max Temperature (K)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['vpd'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max VPD (hPa)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['wind'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max Wind (m/s)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_merra['hd0w0'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max HD0W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd1w0'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max HD1W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd2w0'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max HD2W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd3w0'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max HD3W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd4w0'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max HD4W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd5w0'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Max HD5W0', 'CO (kg)'], 1)

plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd0w0_MA'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['HD0W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd1w0_MA'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['HD1W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd2w0_MA'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['HD2W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd3w0_MA'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['HD3W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd4w0_MA'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['HD4W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd5w0_MA'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['HD5W0 Moving', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_ncar['mlive']*100, df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Live FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_ncar['mdead']*100, df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Dead FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_personnel['personnel'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['Personnel (#)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_pws['pws'], df_rave['CO']]), 'Bobcat Polygons',\
             [], ['PWS', 'CO (kg)'], 1)

# %%
#concatenate all the dataframes
df_all = pd.concat([df_merra.drop(columns=['Unnamed: 0', 'frp', 'area', 'num points']),\
                    df_movingaverage.drop(columns=['Unnamed: 0','temp','vpd','wind', 'frp', 'area', 'num points']),\
                    #df_ncar.drop(columns=['Unnamed: 0']),\
                    #df_personnel.drop(columns=['Unnamed: 0']),\
                    df_pws.drop(columns=['Unnamed: 0']),\
                    df_rave['CO']], axis=1)
df_all
corrMatrix = df_all.corr()
fig,ax=plt.subplots(figsize=(18,18))
sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, cmap='seismic')
plt.title('Bobcat Active Days')
plt.show()

# %% [markdown]
# ## Holiday Farm

# %%
df_merra = pd.read_csv('HOLIDAY_FARM_Daily_MERRA.csv')
df_rave = pd.read_csv('HOLIDAY_FARM_Daily_RAVE.csv')
df_ncar = pd.read_csv('HOLIDAY_FARM_Daily_NCAR_Moisture.csv')
df_personnel = pd.read_csv('HOLIDAY_FARM_Daily_Personnel.csv')
df_pws = pd.read_csv('HOLIDAY_FARM_Daily_PWS.csv')

# %%
plot_twoline(df_merra['day'], np.stack([df_merra['temp'], df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Max Temperature (K)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['vpd'], df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Max VPD (hPa)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['wind'], df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Max Wind (m/s)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_merra['hd0w0'], df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Max HD0W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd1w0'], df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Max HD1W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd2w0'], df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Max HD2W0', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_ncar['mlive']*100, df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Live FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_ncar['mdead']*100, df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Dead FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_personnel['personnel'], df_rave['CO']]), 'Holiday Farm Thapa Polygons',\
             [], ['Personnel (#)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_pws['pws'], df_rave['CO']]), 'Holiday Farm  Thapa Polygons',\
             [], ['PWS', 'CO (kg)'], 1)

# %%
#concatenate all the dataframes
df_all = pd.concat([df_merra.drop(columns=['Unnamed: 0', 'frp', 'area', 'num points']),\
                    df_ncar.drop(columns=['Unnamed: 0']),\
                    df_personnel.drop(columns=['Unnamed: 0']),\
                    df_pws.drop(columns=['Unnamed: 0']),\
                    df_rave['CO']], axis=1)
df_all
corrMatrix = df_all.corr()
fig,ax=plt.subplots(figsize=(13,13))
sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, cmap='seismic')
plt.title('Correlations for Holiday Farm All Days')
plt.show()

# %%
# find the most active days


plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_merra['temp'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Max Temperature (K)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_merra['vpd'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Max VPD (hPa)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_merra['wind'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Max Wind (m/s)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_merra['hd0w0'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Max HD0W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_merra['hd1w0'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Max HD1W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_merra['hd2w0'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Max HD2W0', 'CO (kg)'], 1)

plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_ncar['mlive'].iloc[0:10]*100, df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Live FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_ncar['mdead'].iloc[0:10]*100, df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Dead FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_personnel['personnel'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['Personnel (#)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'].iloc[0:10], np.stack([df_pws['pws'].iloc[0:10], df_rave['CO'].iloc[0:10]]), 'Holiday Farm Thapa Polygons',\
             [], ['PWS', 'CO (kg)'], 1)

# %%
#concatenate all the dataframes
df_all = pd.concat([df_merra.iloc[0:10].drop(columns=['Unnamed: 0', 'frp', 'area', 'num points']),\
                    df_ncar.iloc[0:10].drop(columns=['Unnamed: 0']),\
                    df_personnel.iloc[0:10].drop(columns=['Unnamed: 0']),\
                    df_pws.iloc[0:10].drop(columns=['Unnamed: 0']),\
                    df_rave['CO'].iloc[0:10]], axis=1)
corrMatrix = df_all.corr()
fig,ax=plt.subplots(figsize=(13,13))
sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, cmap='seismic')
plt.title('Correlations For Holiday Farm Active Days', fontsize=20)

plt.show()

# %% [markdown]
# ## Dolan

# %%
df_merra = pd.read_csv('DOLAN_Daily_MERRA.csv').iloc[0:30]
df_rave = pd.read_csv('DOLAN_Daily_RAVE.csv').iloc[0:30]
df_ncar = pd.read_csv('DOLAN_Daily_NCAR_Moisture.csv').iloc[0:30]
df_personnel = pd.read_csv('DOLAN_Daily_Personnel.csv').iloc[0:30]
df_pws = pd.read_csv('DOLAN_Daily_PWS.csv').iloc[0:30]
df_movingaverage = pd.read_csv('DOLAN_Daily_MERRA_movingaverage.csv').iloc[0:30]


# %%
plot_twoline(df_merra['day'], np.stack([df_merra['temp'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max Temperature (K)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['vpd'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max VPD (hPa)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['wind'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max Wind (m/s)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_merra['hd0w0'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max HD0W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd1w0'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max HD1W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd2w0'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max HD2W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd3w0'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max HD3W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd4w0'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max HD4W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_merra['hd5w0'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Max HD5W0', 'CO (kg)'], 1)

plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd0w0_MA'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['HD0W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd1w0_MA'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['HD1W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd2w0_MA'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['HD2W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd3w0_MA'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['HD3W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd4w0_MA'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['HD4W0 Moving', 'CO (kg)'], 1)
plot_twoline(df_movingaverage['day'], np.stack([df_movingaverage['hd5w0_MA'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['HD5W0 Moving', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_ncar['mlive']*100, df_rave['CO']]), 'Dolan Polygons',\
             [], ['Live FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_ncar['mdead']*100, df_rave['CO']]), 'Dolan Polygons',\
             [], ['Dead FMC (%)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_personnel['personnel'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['Personnel (#)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'], np.stack([df_pws['pws'], df_rave['CO']]), 'Dolan Polygons',\
             [], ['PWS', 'CO (kg)'], 1)

# %%
#concatenate all the dataframes
df_all = pd.concat([df_merra.drop(columns=['Unnamed: 0', 'frp', 'area', 'num points']),\
                    df_movingaverage.drop(columns=['Unnamed: 0','temp','vpd','wind', 'frp', 'area', 'num points']),\
                    #df_ncar.drop(columns=['Unnamed: 0']),\
                    #df_personnel.drop(columns=['Unnamed: 0']),\
                    df_pws.drop(columns=['Unnamed: 0']),\
                    df_rave['CO']], axis=1)
df_all
corrMatrix = df_all.corr()
fig,ax=plt.subplots(figsize=(18,18))
sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, cmap='seismic')
plt.title('Dolan Active Days')
plt.show()

# %%


# %% [markdown]
# ## Look at all 3 fires

# %%
df_merra = pd.concat([pd.read_csv('AC_Daily_MERRA.csv').iloc[1:50], pd.read_csv('BOBCAT_Daily_MERRA.csv').iloc[1:18],pd.read_csv('HOLIDAY_FARM_Daily_MERRA.csv').iloc[1:10]], axis=0, ignore_index=True)
df_rave = pd.concat([pd.read_csv('AC_Daily_RAVE.csv').iloc[1:50], pd.read_csv('BOBCAT_Daily_RAVE.csv').iloc[1:18],pd.read_csv('HOLIDAY_FARM_Daily_RAVE.csv').iloc[1:10]], axis=0, ignore_index=True)
df_ncar = pd.concat([pd.read_csv('AC_Daily_NCAR_Moisture.csv').iloc[1:50], pd.read_csv('BOBCAT_Daily_NCAR_Moisture.csv').iloc[1:18],pd.read_csv('HOLIDAY_FARM_Daily_NCAR_Moisture.csv').iloc[1:10]], axis=0, ignore_index=True)
df_personnel = pd.concat([pd.read_csv('AC_Daily_Personnel.csv').iloc[1:50], pd.read_csv('BOBCAT_Daily_Personnel.csv').iloc[1:18],pd.read_csv('HOLIDAY_FARM_Daily_Personnel.csv').iloc[1:10]], axis=0, ignore_index=True)
df_pws = pd.concat([pd.read_csv('AC_Daily_PWS.csv').iloc[1:50], pd.read_csv('BOBCAT_Daily_PWS.csv').iloc[1:18],pd.read_csv('HOLIDAY_FARM_Daily_PWS.csv').iloc[1:10]], axis=0, ignore_index=True)

# %%
df_merra

# %%
#concatenate all the dataframes
df_all = pd.concat([df_merra.drop(columns=['Unnamed: 0', 'frp', 'area', 'num points']),\
                    df_ncar.drop(columns=['Unnamed: 0']),\
                    df_personnel.drop(columns=['Unnamed: 0']),\
                    df_pws.drop(columns=['Unnamed: 0']),\
                    df_rave['CO']], axis=1)
df_all
corrMatrix = df_all.corr()
fig,ax=plt.subplots(figsize=(13,13))
sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, cmap='seismic')
plt.show()

# %% [markdown]
# ## Make MERRA Time Series Plots

# %%
plot_twoline(df_merra['day'].iloc[0:69], np.stack([df_merra['temp'].iloc[0:69], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Max Temperature (K)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:69], np.stack([df_merra['vpd'].iloc[0:69], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Max VPD (hPa)', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:69], np.stack([df_merra['wind'].iloc[0:69], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Max Wind (m/s)', 'CO (kg)'], 1)

plot_twoline(df_merra['day'].iloc[0:69], np.stack([df_merra['hd0w0'].iloc[0:69], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Max HD0W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:69], np.stack([df_merra['hd1w0'].iloc[0:69], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Max HD1W0', 'CO (kg)'], 1)
plot_twoline(df_merra['day'].iloc[0:69], np.stack([df_merra['hd2w0'].iloc[0:69], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Max HD2W0', 'CO (kg)'], 1)

# %% [markdown]
# ## Make FCCS time Series Plots

# %%
print(df_fccs['fuels'].values)
plot_twoline(df_merra['day'], np.stack([df_fccs['fuels'], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Mode Fuel', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_fccs['slopes'], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Max Slope', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_fccs['aspects'], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Mode Aspect', 'CO (kg)'], 1)

# %% [markdown]
# ## Make NCAR Fuel Moisture Time Series

# %%
plot_twoline(df_merra['day'], np.stack([df_ncar['mlive'], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Live Fuel Moisure', 'CO (kg)'], 1)
plot_twoline(df_merra['day'], np.stack([df_ncar['mdead'], df_rave['CO']]), 'August Complex Thapa Polygons',\
             [], ['Dead Fuel Moisure', 'CO (kg)'], 1)

# %% [markdown]
# ## Personnel

# %%


# %% [markdown]
# ## Make Corellograms

# %%


# %%


# %% [markdown]
# ## Helper Functions





# %%




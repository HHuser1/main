# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:11:59 2022

@author: HH
"""

import netCDF4 as nc
from scipy.stats import pearsonr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.util import add_cyclic_point 
import matplotlib.path as mpath
from cartopy.mpl import geoaxes
import matplotlib.ticker as mticker
from matplotlib import patches
from scipy.stats.mstats import ttest_ind
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cmaps

f1=xr.open_dataset('D:/data/ERA5/total precipitation1951-2021.nc')
pre_cli=np.array(f1.tp.loc['1979-01-01':'2021-12-01',45:35,0:30]).reshape(43,12,41,121).mean((0,2,3))
pre_c=pre_cli*30*1000

pre_cli1=(np.array(f1.tp.loc['1979-01-01':'2021-12-01',45:35,0:30]).reshape(43,12,41,121).mean((2,3)))*30*1000
pre_std=np.std(pre_cli1,axis=0)

font={'family':'Arial'} 
lat=np.array(f1.latitude.loc[80:0])
lon=np.array(f1.longitude.loc[:])
t=np.arange(1,13,1)
fig = plt.figure(figsize=(12,12))
ax = fig.add_axes([0.15, 0.2, 0.5, 0.2])
ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],fontsize=14,fontdict=font)
#ax.set_ylabel('precipitation (mm)',fontsize=14,fontdict=font)
ax.tick_params(length=5,width=1,labelsize=12)
ax.set_ylim(0,120)
ax.set_yticks([0,20,40,60,80,100,120])
ax.set_xlabel('Month',fontdict=font,fontsize=13)
ax.set_ylabel('Precipitation(mm)',fontdict=font,fontsize=13)
ax.bar(t,pre_c,zorder=1,alpha=0.5,color='darkorange',width=0.5)
ax.errorbar(t,pre_c,xerr=None,yerr=pre_std,ls='none',ecolor='k',capsize=6)
#ax.set_title('Climatological precipitation',loc='right',fontproperties='Arial',fontsize=18)



#plt.savefig('D:/figure/改文章6/地中海气候态降水.png',format='png',bbox_inches='tight',dpi=500)











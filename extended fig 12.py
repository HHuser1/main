# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:16:55 2023

@author: HH
"""

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
from scipy import signal
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.stats import zscore
import pandas as pd
from scipy.stats.mstats import ttest_ind
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cmaps
from scipy.stats import linregress


f1=xr.open_dataset('D:/data/FAMIPC5_clim_VerticalTransfer_Z3_monthly.nc')
f2=xr.open_dataset('D:/data/FAMIPC5_NA_REALwarm_VerticalTransfer_Z3_monthly.nc')
f3=xr.open_dataset('D:/data/FAMIPC5_Med_REALwarm_VerticalTransfer_Z3_monthly.nc')
f4=xr.open_dataset('D:/data/FAMIPC5_inputdata_forcing_climatol_1x1_2.nc')
f5=xr.open_dataset('D:/data/FAMIPC5_inputdata_REALforcing_NA_1x1.nc')
f6=xr.open_dataset('D:/data/FAMIPC5_inputdata_REALforcing_Med_1x1_2.nc')

lat=np.array(f2.lat.loc[-1:90])
lon=np.array(f2.lon.loc[:])

lat1=np.array(f4.lat.loc[0:90])
lon1=np.array(f4.lon.loc[:])

z_c=np.array(f1.Z3.loc[10:11,300,-1:90,:])
z_na=np.array(f2.Z3.loc[10:11,300,-1:90,:])
z_na_ano=z_na-z_c
z_na_ano,cyclic_lons=add_cyclic_point(z_na_ano,coord=lon)


z_med=np.array(f3.Z3.loc[10:11,300,-1:90,:])
z_med_ano=z_med-z_c
z_med_ano,cyclic_lons=add_cyclic_point(z_med_ano,coord=lon)

sst_c=np.array(f4.SST_cpl.loc['1990-11-16',0:90,:])
sst_na=np.array(f5.SST_cpl.loc['1990-11-16',0:90,:])
sst_na_ano=(sst_na-sst_c).reshape(90,360)
a=np.isnan(sst_na_ano)
for i in range(0,90):
    for j in range(0,360):
        if a[i,j]==True:
            sst_na_ano[i,j]=0
sst_na_ano,cyclic_lons1=add_cyclic_point(sst_na_ano,coord=lon1)


sst_med=np.array(f6.SST_cpl.loc['1990-11-16',0:90,:])
sst_med_ano=(sst_med-sst_c).reshape(90,360)
a=np.isnan(sst_med_ano)
for i in range(0,90):
    for j in range(0,360):
        if a[i,j]==True:
            sst_med_ano[i,j]=0
sst_med_ano,cyclic_lons1=add_cyclic_point(sst_med_ano,coord=lon1)

cmap=cmaps.BlueWhiteOrangeRed 
cmap1=cmaps.BlueDarkRed18

fig = plt.figure(figsize=(12,8))
proj = ccrs.PlateCarree(central_longitude=15)
proj2 = ccrs.PlateCarree(central_longitude=-15)
proj1 = ccrs.PlateCarree(central_longitude=120)

font={'family':'Arial'}
ax = fig.add_axes([0.1, 0.6, 0.4, 0.4],projection = proj)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=1)#线宽lw
ax.add_feature(cfeature.LAND,color='gray')#给陆地填色
ax.set_extent([-30,60,20,60],crs=ccrs.PlateCarree())
ax.set_xticks(np.arange(-30,90,30),crs=ccrs.PlateCarree())
ax.set_yticks(np.array([20,40,60]),crs=ccrs.PlateCarree())
ax.set_xticklabels(np.arange(-30,90,30),fontdict=font)
ax.set_yticklabels([20,40,60],fontdict=font)
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c1=ax.contourf(cyclic_lons1,lat1,sst_med_ano,transform=ccrs.PlateCarree(),cmap=cmap,zorder=0,levels=np.arange(-1.3,1.4,0.2),extend='both')
ax.tick_params(labelsize=14,width=1.5,length=6)
ax.set_title('(a) Exp_Med',loc='left',fontsize=16,fontproperties='Arial')
ax.set_title('Nov',loc='center',fontsize=16,fontproperties='Arial')
ax.set_title('SST',loc='right',fontsize=16,fontproperties='Arial')



ax1 = fig.add_axes([0.56, 0.61, 0.38, 0.38],projection = proj2)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=1)#线宽lw
ax1.add_feature(cfeature.LAND,color='gray')#给陆地填色
ax1.set_extent([-120,30,10,80],crs=ccrs.PlateCarree())
ax1.set_xticks(np.arange(-120,60,30),crs=ccrs.PlateCarree())
ax1.set_yticks(np.array([20,40,60,80]),crs=ccrs.PlateCarree())
ax1.set_xticklabels(np.arange(-120,60,30),fontdict=font)
ax1.set_yticklabels([20,40,60,80],fontdict=font)
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
#c1=ax1.contourf(cyclic_lons1,lat1,sst_na_ano,transform=ccrs.PlateCarree(),cmap='RdBu_r',zorder=0,levels=np.arange(-0.7,0.71,0.1),extend='both')
c1=ax1.contourf(cyclic_lons1,lat1,sst_na_ano,transform=ccrs.PlateCarree(),levels=np.arange(-1.3,1.4,0.2),zorder=0,cmap=cmap,extend='both')

ax1.tick_params(labelsize=14,width=1.5,length=6)
ax1.set_title('(d) Exp_NAtl',loc='left',fontsize=16,fontproperties='Arial')
ax1.set_title('Nov',loc='center',fontsize=16,fontproperties='Arial')
ax1.set_title('SST',loc='right',fontsize=16,fontproperties='Arial')


position=fig.add_axes([0.965,0.68,0.01,0.25])
clb=plt.colorbar(c1,cax=position,orientation='vertical',format='%.1f',ticks=[-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6])
clb.ax.tick_params(labelsize=14,width=1.5)


ax2 = fig.add_axes([0.095, 0.3, 0.4, 0.4],projection = proj1)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=1)#线宽lw
ax2.add_feature(cfeature.LAND,color='gray')#给陆地填色
#ax2.set_extent([-60,60,0,80],crs=ccrs.PlateCarree())
ax2.set_xticks(np.arange(-60,360,60),crs=ccrs.PlateCarree())
ax2.set_yticks(np.array([0,30,60,90]),crs=ccrs.PlateCarree())
ax2.set_xticklabels(np.arange(-60,360,60),fontdict=font)
ax2.set_yticklabels([0,30,60,90],fontdict=font)
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c1=ax2.contourf(cyclic_lons,lat,z_med_ano[0],transform=ccrs.PlateCarree(),cmap=cmap,zorder=0,levels=np.arange(-60,70,10),extend='both')
ax2.tick_params(labelsize=14,width=1.5,length=6)
ax2.set_title('(b) Exp_Med',loc='left',fontsize=16,fontproperties='Arial')
ax2.set_title('Oct',loc='center',fontsize=16,fontproperties='Arial')
ax2.set_title('500hPa HGT',loc='right',fontsize=16,fontproperties='Arial')


ax3 = fig.add_axes([0.555, 0.3, 0.4, 0.4],projection = proj1)
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=1)#线宽lw
ax3.add_feature(cfeature.LAND,color='gray')#给陆地填色
#ax3.set_extent([-60,60,0,80],crs=ccrs.PlateCarree())
ax3.set_xticks(np.arange(-60,360,60),crs=ccrs.PlateCarree())
ax3.set_yticks(np.array([0,30,60,90]),crs=ccrs.PlateCarree())
ax3.set_xticklabels(np.arange(-60,360,60),fontdict=font)
ax3.set_yticklabels([0,30,60,90],fontdict=font)
ax3.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax3.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c1=ax3.contourf(cyclic_lons,lat,z_na_ano[0],transform=ccrs.PlateCarree(),cmap=cmap,zorder=0,levels=np.arange(-60,70,10),extend='both')
ax3.tick_params(labelsize=14,width=1.5,length=6)
ax3.set_title('(e) Exp_NAtl',loc='left',fontsize=16,fontproperties='Arial')
ax3.set_title('Oct',loc='center',fontsize=16,fontproperties='Arial')
ax3.set_title('500hPa HGT',loc='right',fontsize=16,fontproperties='Arial')




ax4 = fig.add_axes([0.095, 0.055, 0.4, 0.4],projection = proj1)
ax4.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=1)#线宽lw
ax4.add_feature(cfeature.LAND,color='gray')#给陆地填色
#ax4.set_extent([-60,60,0,80],crs=ccrs.PlateCarree())
ax4.set_xticks(np.arange(-60,360,60),crs=ccrs.PlateCarree())
ax4.set_yticks(np.array([0,30,60,90]),crs=ccrs.PlateCarree())
ax4.set_xticklabels(np.arange(-60,360,60),fontdict=font)
ax4.set_yticklabels([0,30,60,90],fontdict=font)
ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c1=ax4.contourf(cyclic_lons,lat,z_med_ano[1],transform=ccrs.PlateCarree(),cmap=cmap,zorder=0,levels=np.arange(-60,70,10),extend='both')
ax4.tick_params(labelsize=14,width=1.5,length=6)
ax4.set_title('(c) Exp_Med',loc='left',fontsize=16,fontproperties='Arial')
ax4.set_title('Nov',loc='center',fontsize=16,fontproperties='Arial')
ax4.set_title('500hPa HGT',loc='right',fontsize=16,fontproperties='Arial')


ax5 = fig.add_axes([0.555, 0.055, 0.4, 0.4],projection = proj1)
ax5.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=1)#线宽lw
ax5.add_feature(cfeature.LAND,color='gray')#给陆地填色
#ax5.set_extent([-60,60,0,80],crs=ccrs.PlateCarree())
ax5.set_xticks(np.arange(-60,360,60),crs=ccrs.PlateCarree())
ax5.set_yticks(np.array([0,30,60,90]),crs=ccrs.PlateCarree())
ax5.set_xticklabels(np.arange(-60,360,60),fontdict=font)
ax5.set_yticklabels([0,30,60,90],fontdict=font)
ax5.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax5.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c1=ax5.contourf(cyclic_lons,lat,z_na_ano[1],transform=ccrs.PlateCarree(),cmap=cmap,zorder=0,levels=np.arange(-60,70,10),extend='both')
ax5.tick_params(labelsize=14,width=1.5,length=6)
ax5.set_title('(f) Exp_NAtl',loc='left',fontsize=16,fontproperties='Arial')
ax5.set_title('Nov',loc='center',fontsize=16,fontproperties='Arial')
ax5.set_title('500hPa HGT',loc='right',fontsize=16,fontproperties='Arial')

position=fig.add_axes([0.965,0.23,0.01,0.3])
clb=plt.colorbar(c1,cax=position,orientation='vertical',format='%.0f')
clb.ax.tick_params(labelsize=14,width=1.5)



#plt.savefig('D:/figure/改文章6/FAMIPC5sst和hgt组图.png',bbox_inches='tight',dpi=300)














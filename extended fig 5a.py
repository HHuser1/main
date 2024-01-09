# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:22:11 2023

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
def sp(str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,str11,str12,str13,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13):
    pre_cli=np.array(f1.tp.loc['1979-01-01':'2021-12-01',80:0,:]).reshape(43,12,321,1440).mean((0))
    e1=np.array(f1.tp.loc[str1,80:0,:])
    e2=np.array(f1.tp.loc[str2,80:0,:])
    e3=np.array(f1.tp.loc[str3,80:0,:])
    e4=np.array(f1.tp.loc[str4,80:0,:])
    e5=np.array(f1.tp.loc[str5,80:0,:])
    e6=np.array(f1.tp.loc[str6,80:0,:])
    e7=np.array(f1.tp.loc[str7,80:0,:])
    e8=np.array(f1.tp.loc[str8,80:0,:])
    e9=np.array(f1.tp.loc[str9,80:0,:])
    e10=np.array(f1.tp.loc[str10,80:0,:])
    e11=np.array(f1.tp.loc[str11,80:0,:])
    e12=np.array(f1.tp.loc[str12,80:0,:])
    e13=np.array(f1.tp.loc[str13,80:0,:])
    
    e_act=(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13)/13
    

    e1_ano=(e1-pre_cli[n1,:,:]).reshape(1,321,1440)
    e2_ano=(e2-pre_cli[n2,:,:]).reshape(1,321,1440)
    e3_ano=(e3-pre_cli[n3,:,:]).reshape(1,321,1440)
    e4_ano=(e4-pre_cli[n4,:,:]).reshape(1,321,1440)
    e5_ano=(e5-pre_cli[n5,:,:]).reshape(1,321,1440)
    e6_ano=(e6-pre_cli[n6,:,:]).reshape(1,321,1440)
    e7_ano=(e7-pre_cli[n7,:,:]).reshape(1,321,1440)
    e8_ano=(e8-pre_cli[n8,:,:]).reshape(1,321,1440)
    e9_ano=(e9-pre_cli[n9,:,:]).reshape(1,321,1440)
    e10_ano=(e10-pre_cli[n10,:,:]).reshape(1,321,1440)
    e11_ano=(e11-pre_cli[n11,:,:]).reshape(1,321,1440)
    e12_ano=(e12-pre_cli[n12,:,:]).reshape(1,321,1440)
    e13_ano=(e13-pre_cli[n13,:,:]).reshape(1,321,1440)

    e_ano=np.concatenate((e1_ano,e2_ano,e3_ano,e4_ano,e5_ano,e6_ano,e7_ano,e8_ano,e9_ano,e10_ano,e11_ano,e12_ano,e13_ano),axis=0)
    e_ano_mean=e_ano.mean((0))
    cyclic_data,cyclic_lons=add_cyclic_point(e_ano_mean,coord=lon)
    a=np.zeros((321,1440))
    t,p=ttest_ind(e_ano,a,equal_var=False)
    cyclic_p,cyclic_lons=add_cyclic_point(p,coord=lon)
    
    return cyclic_data,cyclic_p,cyclic_lons



lat=np.array(f1.latitude.loc[80:0])
lon=np.array(f1.longitude.loc[:])

fig = plt.figure(figsize=(12,12))
proj = ccrs.PlateCarree(central_longitude=0)


cyclic_data,cyclic_p,cyclic_lons=sp('1956-10-01','1957-10-01','1990-10-01','2013-10-01','2014-10-01','1961-10-01','1962-10-01','1985-10-01','1989-10-01','1993-10-01','2004-10-01','2019-10-01','2020-10-01',9,9,9,9,9,9,9,9,9,9,9,9,9)
u1,v1=np.zeros((321,1440)),np.zeros((321,1440))
font={'family':'Arial'}
ax2 = fig.add_axes([0.3, 0.75, 0.6, 0.6],projection = proj)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax2.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax2.set_extent([-30,150,0,80],crs=ccrs.PlateCarree())
ax2.set_xticks(np.array([-30,0,30,60,90,120,150]),crs=ccrs.PlateCarree())
ax2.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax2.set_xticklabels([-30,0,30,60,90,120,150],fontdict=font)
ax2.set_yticklabels([0,20,40,60,80],fontdict=font)
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax2.contourf(cyclic_lons,lat,cyclic_data*30*1000,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-30,36,6),zorder=0,cmap='BrBG')
d=patches.Rectangle((0,35),30,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax2.add_patch(d)

d=patches.Rectangle((334,48),26,17,linewidth=2,linestyle='-',zorder=4,edgecolor='k',facecolor='none',transform=ccrs.PlateCarree())
ax2.add_patch(d)

ax2.tick_params(axis='x',labelsize=14,width=1.5,length=7)
ax2.tick_params(axis='y',labelsize=14,width=1.5,length=7)


ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(True)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['right'].set_linewidth(1.5)
ax2.spines['top'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)
#ax2.set_title('(a)',loc='left',fontsize=16,fontproperties='Arial')
#ax2.set_title('(a) Nov',loc='left',fontsize=16,fontproperties='Arial')
#ax2.set_title('Precipitation',loc='right',fontsize=16,fontproperties='Arial')
for i in range(0,321,10):
    for j in range(0,1440,10):
        if j<=720 and cyclic_p[i,j]<=0.1:
            ax2.plot(lon[j],lat[i],marker='o',markersize=1,c='r')
        if j>720 and cyclic_p[i,j]<=0.1:
            ax2.plot(lon[j]-360,lat[i],marker='o',markersize=1,c='r')

for i in range(0,321,5):
    for j in range(0,1440,5):
        if 140<=i<=180 and 0<=j<=120 and cyclic_p[i,j]<=0.1:
            ax2.plot(lon[j],lat[i],marker='o',markersize=1,c='r')
position=fig.add_axes([0.91,0.915,0.015,0.27])
clb=plt.colorbar(c2,cax=position,orientation='vertical',format='%.0f',ticks=[-30,-24,-18,-12,-6,0,6,12,18,24,30])
clb.ax.tick_params(labelsize=14,width=1.5)  



plt.savefig('D:/figure/改文章6/10月降水.pdf',dpi=300,bbox_inches='tight')  
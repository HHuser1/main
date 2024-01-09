# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 19:27:35 2023

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
from matplotlib import patheffects

f1=xr.open_dataset('D:/data/ERA5/geopotential 1950-1958.nc')
f2=xr.open_dataset('D:/data/ERA5/geopotential 1959-2021.nc')

f3=xr.open_dataset('D:/data/ERA5/u10 v10 slp 1950-1958.nc')
f4=xr.open_dataset('D:/data/ERA5/u10 v10 slp 1959-2021.nc')
pre_cli_300=np.array(f2.z.loc['1979-01-01':'2021-12-01',300,80:0,:]).reshape(43,12,321,1440).mean((0))
pre_cli_500=np.array(f4.msl.loc['1979-01-01':'2021-12-01',80:0,:]).reshape(43,12,321,1440).mean((0))
pre_cli_850=np.array(f2.z.loc['1979-01-01':'2021-12-01',850,80:0,:]).reshape(43,12,321,1440).mean((0))




def sp500(str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,str11,str12,str13,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13):
    e1=np.array(f3.msl.loc[str1,80:0,:])
    e2=np.array(f3.msl.loc[str2,80:0,:])
    e3=np.array(f4.msl.loc[str3,80:0,:])
    e4=np.array(f4.msl.loc[str4,80:0,:])
    e5=np.array(f4.msl.loc[str5,80:0,:])
    e6=np.array(f4.msl.loc[str6,80:0,:])
    e7=np.array(f4.msl.loc[str7,80:0,:])
    e8=np.array(f4.msl.loc[str8,80:0,:])
    e9=np.array(f4.msl.loc[str9,80:0,:])
    e10=np.array(f4.msl.loc[str10,80:0,:])
    e11=np.array(f4.msl.loc[str11,80:0,:])
    e12=np.array(f4.msl.loc[str12,80:0,:])
    e13=np.array(f4.msl.loc[str13,80:0,:])
    
    e_act=(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13)/13
    

    e1_ano=(e1-pre_cli_500[n1,:,:]).reshape(1,321,1440)
    e2_ano=(e2-pre_cli_500[n2,:,:]).reshape(1,321,1440)
    e3_ano=(e3-pre_cli_500[n3,:,:]).reshape(1,321,1440)
    e4_ano=(e4-pre_cli_500[n4,:,:]).reshape(1,321,1440)
    e5_ano=(e5-pre_cli_500[n5,:,:]).reshape(1,321,1440)
    e6_ano=(e6-pre_cli_500[n6,:,:]).reshape(1,321,1440)
    e7_ano=(e7-pre_cli_500[n7,:,:]).reshape(1,321,1440)
    e8_ano=(e8-pre_cli_500[n8,:,:]).reshape(1,321,1440)
    e9_ano=(e9-pre_cli_500[n9,:,:]).reshape(1,321,1440)
    e10_ano=(e10-pre_cli_500[n10,:,:]).reshape(1,321,1440)
    e11_ano=(e11-pre_cli_500[n11,:,:]).reshape(1,321,1440)
    e12_ano=(e12-pre_cli_500[n12,:,:]).reshape(1,321,1440)
    e13_ano=(e13-pre_cli_500[n13,:,:]).reshape(1,321,1440)

    e_ano=np.concatenate((e1_ano,e2_ano,e3_ano,e4_ano,e5_ano,e6_ano,e7_ano,e8_ano,e9_ano,e10_ano,e11_ano,e12_ano,e13_ano),axis=0)
    e_ano_mean=e_ano.mean((0))
    a=np.zeros((321,1440))
    t,p=ttest_ind(e_ano,a,equal_var=False)
    e_ano_mean,cyclic_lons=add_cyclic_point(e_ano_mean,coord=lon)
    p,cyclic_lons=add_cyclic_point(p,coord=lon)
    
    
    return e_ano_mean,p,cyclic_lons
def sp300(str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,str11,str12,str13,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13):
    e1=np.array(f1.z.loc[str1,300,80:0,:])
    e2=np.array(f1.z.loc[str2,300,80:0,:])
    e3=np.array(f2.z.loc[str3,300,80:0,:])
    e4=np.array(f2.z.loc[str4,300,80:0,:])
    e5=np.array(f2.z.loc[str5,300,80:0,:])
    e6=np.array(f2.z.loc[str6,300,80:0,:])
    e7=np.array(f2.z.loc[str7,300,80:0,:])
    e8=np.array(f2.z.loc[str8,300,80:0,:])
    e9=np.array(f2.z.loc[str9,300,80:0,:])
    e10=np.array(f2.z.loc[str10,300,80:0,:])
    e11=np.array(f2.z.loc[str11,300,80:0,:])
    e12=np.array(f2.z.loc[str12,300,80:0,:])
    e13=np.array(f2.z.loc[str13,300,80:0,:])
    
    e_act=(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13)/13
    

    e1_ano=(e1-pre_cli_300[n1,:,:]).reshape(1,321,1440)
    e2_ano=(e2-pre_cli_300[n2,:,:]).reshape(1,321,1440)
    e3_ano=(e3-pre_cli_300[n3,:,:]).reshape(1,321,1440)
    e4_ano=(e4-pre_cli_300[n4,:,:]).reshape(1,321,1440)
    e5_ano=(e5-pre_cli_300[n5,:,:]).reshape(1,321,1440)
    e6_ano=(e6-pre_cli_300[n6,:,:]).reshape(1,321,1440)
    e7_ano=(e7-pre_cli_300[n7,:,:]).reshape(1,321,1440)
    e8_ano=(e8-pre_cli_300[n8,:,:]).reshape(1,321,1440)
    e9_ano=(e9-pre_cli_300[n9,:,:]).reshape(1,321,1440)
    e10_ano=(e10-pre_cli_300[n10,:,:]).reshape(1,321,1440)
    e11_ano=(e11-pre_cli_300[n11,:,:]).reshape(1,321,1440)
    e12_ano=(e12-pre_cli_300[n12,:,:]).reshape(1,321,1440)
    e13_ano=(e13-pre_cli_300[n13,:,:]).reshape(1,321,1440)

    e_ano=np.concatenate((e1_ano,e2_ano,e3_ano,e4_ano,e5_ano,e6_ano,e7_ano,e8_ano,e9_ano,e10_ano,e11_ano,e12_ano,e13_ano),axis=0)
    e_ano_mean=e_ano.mean((0))
    a=np.zeros((321,1440))
    t,p=ttest_ind(e_ano,a,equal_var=False)
    e_ano_mean,cyclic_lons=add_cyclic_point(e_ano_mean,coord=lon)
    p,cyclic_lons=add_cyclic_point(p,coord=lon)
    
    return e_ano_mean,p,cyclic_lons

def sp850(str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,str11,str12,str13,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13):
    e1=np.array(f1.z.loc[str1,850,80:0,:])
    e2=np.array(f1.z.loc[str2,850,80:0,:])
    e3=np.array(f2.z.loc[str3,850,80:0,:])
    e4=np.array(f2.z.loc[str4,850,80:0,:])
    e5=np.array(f2.z.loc[str5,850,80:0,:])
    e6=np.array(f2.z.loc[str6,850,80:0,:])
    e7=np.array(f2.z.loc[str7,850,80:0,:])
    e8=np.array(f2.z.loc[str8,850,80:0,:])
    e9=np.array(f2.z.loc[str9,850,80:0,:])
    e10=np.array(f2.z.loc[str10,850,80:0,:])
    e11=np.array(f2.z.loc[str11,850,80:0,:])
    e12=np.array(f2.z.loc[str12,850,80:0,:])
    e13=np.array(f2.z.loc[str13,850,80:0,:])
    
    e_act=(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13)/13
    

    e1_ano=(e1-pre_cli_850[n1,:,:]).reshape(1,321,1440)
    e2_ano=(e2-pre_cli_850[n2,:,:]).reshape(1,321,1440)
    e3_ano=(e3-pre_cli_850[n3,:,:]).reshape(1,321,1440)
    e4_ano=(e4-pre_cli_850[n4,:,:]).reshape(1,321,1440)
    e5_ano=(e5-pre_cli_850[n5,:,:]).reshape(1,321,1440)
    e6_ano=(e6-pre_cli_850[n6,:,:]).reshape(1,321,1440)
    e7_ano=(e7-pre_cli_850[n7,:,:]).reshape(1,321,1440)
    e8_ano=(e8-pre_cli_850[n8,:,:]).reshape(1,321,1440)
    e9_ano=(e9-pre_cli_850[n9,:,:]).reshape(1,321,1440)
    e10_ano=(e10-pre_cli_850[n10,:,:]).reshape(1,321,1440)
    e11_ano=(e11-pre_cli_850[n11,:,:]).reshape(1,321,1440)
    e12_ano=(e12-pre_cli_850[n12,:,:]).reshape(1,321,1440)
    e13_ano=(e13-pre_cli_850[n13,:,:]).reshape(1,321,1440)

    e_ano=np.concatenate((e1_ano,e2_ano,e3_ano,e4_ano,e5_ano,e6_ano,e7_ano,e8_ano,e9_ano,e10_ano,e11_ano,e12_ano,e13_ano),axis=0)
    e_ano_mean=e_ano.mean((0))
    a=np.zeros((321,1440))
    t,p=ttest_ind(e_ano,a,equal_var=False)
    e_ano_mean,cyclic_lons=add_cyclic_point(e_ano_mean,coord=lon)
    p,cyclic_lons=add_cyclic_point(p,coord=lon)
    
    return e_ano_mean,p,cyclic_lons
cmap=cmaps.BlueWhiteOrangeRed

lat=np.array(f2.latitude.loc[80:0])
lon=np.array(f2.longitude.loc[:])

font={'family':'Arial'}
fig = plt.figure(figsize=(12,12))
proj = ccrs.PlateCarree(central_longitude=265)
e_ano_mean1_sep,p1_300_sep,cyclic_lons=sp300('1956-09-01','1957-09-01','1990-09-01','2013-09-01','2014-09-01','1961-09-01','1962-09-01','1985-09-01','1989-09-01','1993-09-01','2004-09-01','2019-09-01','2020-09-01',8,8,8,8,8,8,8,8,8,8,8,8,8)
e_ano_mean2_sep,p2_500_sep,cyclic_lons=sp500('1956-09-01','1957-09-01','1990-09-01','2013-09-01','2014-09-01','1961-09-01','1962-09-01','1985-09-01','1989-09-01','1993-09-01','2004-09-01','2019-09-01','2020-09-01',8,8,8,8,8,8,8,8,8,8,8,8,8)
e_ano_mean3_sep,p3_850_sep,cyclic_lons=sp850('1956-09-01','1957-09-01','1990-09-01','2013-09-01','2014-09-01','1961-09-01','1962-09-01','1985-09-01','1989-09-01','1993-09-01','2004-09-01','2019-09-01','2020-09-01',8,8,8,8,8,8,8,8,8,8,8,8,8)

e_ano_mean1_oct,p1_300_oct,cyclic_lons=sp300('1956-10-01','1957-10-01','1990-10-01','2013-10-01','2014-10-01','1961-10-01','1962-10-01','1985-10-01','1989-10-01','1993-10-01','2004-10-01','2019-10-01','2020-10-01',9,9,9,9,9,9,9,9,9,9,9,9,9)
e_ano_mean2_oct,p2_500_oct,cyclic_lons=sp500('1956-10-01','1957-10-01','1990-10-01','2013-10-01','2014-10-01','1961-10-01','1962-10-01','1985-10-01','1989-10-01','1993-10-01','2004-10-01','2019-10-01','2020-10-01',9,9,9,9,9,9,9,9,9,9,9,9,9)
e_ano_mean3_oct,p3_850_oct,cyclic_lons=sp850('1956-10-01','1957-10-01','1990-10-01','2013-10-01','2014-10-01','1961-10-01','1962-10-01','1985-10-01','1989-10-01','1993-10-01','2004-10-01','2019-10-01','2020-10-01',9,9,9,9,9,9,9,9,9,9,9,9,9)

e_ano_mean1_nov,p1_300_nov,cyclic_lons=sp300('1956-11-01','1957-11-01','1990-11-01','2013-11-01','2014-11-01','1961-11-01','1962-11-01','1985-11-01','1989-11-01','1993-11-01','2004-11-01','2019-11-01','2020-11-01',10,10,10,10,10,10,10,10,10,10,10,10,10)
e_ano_mean2_nov,p2_500_nov,cyclic_lons=sp500('1956-11-01','1957-11-01','1990-11-01','2013-11-01','2014-11-01','1961-11-01','1962-11-01','1985-11-01','1989-11-01','1993-11-01','2004-11-01','2019-11-01','2020-11-01',10,10,10,10,10,10,10,10,10,10,10,10,10)
e_ano_mean3_nov,p3_850_nov,cyclic_lons=sp850('1956-11-01','1957-11-01','1990-11-01','2013-11-01','2014-11-01','1961-11-01','1962-11-01','1985-11-01','1989-11-01','1993-11-01','2004-11-01','2019-11-01','2020-11-01',10,10,10,10,10,10,10,10,10,10,10,10,10)

e_ano_mean1_dec,p1_300_dec,cyclic_lons=sp300('1956-12-01','1957-12-01','1990-12-01','2013-12-01','2014-12-01','1961-12-01','1962-12-01','1985-12-01','1989-12-01','1993-12-01','2004-12-01','2019-12-01','2020-12-01',11,11,11,11,11,11,11,11,11,11,11,11,11)
e_ano_mean2_dec,p2_500_dec,cyclic_lons=sp500('1956-12-01','1957-12-01','1990-12-01','2013-12-01','2014-12-01','1961-12-01','1962-12-01','1985-12-01','1989-12-01','1993-12-01','2004-12-01','2019-12-01','2020-12-01',11,11,11,11,11,11,11,11,11,11,11,11,11)
e_ano_mean3_dec,p3_850_dec,cyclic_lons=sp850('1956-12-01','1957-12-01','1990-12-01','2013-12-01','2014-12-01','1961-12-01','1962-12-01','1985-12-01','1989-12-01','1993-12-01','2004-12-01','2019-12-01','2020-12-01',11,11,11,11,11,11,11,11,11,11,11,11,11)

e_ano_mean1_jan,p1_300_jan,cyclic_lons=sp300('1957-01-01','1958-01-01','1991-01-01','2014-01-01','2015-01-01','1962-01-01','1963-01-01','1986-01-01','1990-01-01','1994-01-01','2005-01-01','2020-01-01','2021-01-01',0,0,0,0,0,0,0,0,0,0,0,0,0)
e_ano_mean2_jan,p2_500_jan,cyclic_lons=sp500('1957-01-01','1958-01-01','1991-01-01','2014-01-01','2015-01-01','1962-01-01','1963-01-01','1986-01-01','1990-01-01','1994-01-01','2005-01-01','2020-01-01','2021-01-01',0,0,0,0,0,0,0,0,0,0,0,0,0)
e_ano_mean3_jan,p3_850_jan,cyclic_lons=sp850('1957-01-01','1958-01-01','1991-01-01','2014-01-01','2015-01-01','1962-01-01','1963-01-01','1986-01-01','1990-01-01','1994-01-01','2005-01-01','2020-01-01','2021-01-01',0,0,0,0,0,0,0,0,0,0,0,0,0)



ax = fig.add_axes([0, 0.7, 0.25, 0.25],projection = proj)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax.set_yticklabels([0,20,40,60,80],fontdict=font)
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax.contourf(cyclic_lons,lat,e_ano_mean1_sep/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)

ax.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.set_title('(a) Sep',loc='left',fontsize=15,fontproperties='Arial')
ax.set_title('300hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p1_300_sep[i,j]<0.1:
            ax.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


ax1 = fig.add_axes([0.32, 0.7, 0.25, 0.25],projection = proj)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax1.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax1.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax1.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax1.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax1.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax1.set_yticklabels([0,20,40,60,80],fontdict=font)
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax1.contourf(cyclic_lons,lat,e_ano_mean1_oct/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax1.add_patch(d)

ax1.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax1.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['top'].set_visible(True)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax1.set_title('(d) Oct',loc='left',fontsize=15,fontproperties='Arial')
ax1.set_title('300hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p1_300_oct[i,j]<0.1:
            ax1.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)




ax2 = fig.add_axes([0.64, 0.7, 0.25, 0.25],projection = proj)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax2.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax2.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax2.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax2.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax2.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax2.set_yticklabels([0,20,40,60,80],fontdict=font)
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax2.contourf(cyclic_lons,lat,e_ano_mean1_nov/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax2.add_patch(d)

ax2.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax2.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(True)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['right'].set_linewidth(1.5)
ax2.spines['top'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

ax2.set_title('(g) Nov',loc='left',fontsize=15,fontproperties='Arial')
ax2.set_title('300hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p1_300_nov[i,j]<0.1:
            ax2.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)

ax3 = fig.add_axes([0.96, 0.7, 0.25, 0.25],projection = proj)
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax3.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax3.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax3.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax3.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax3.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax3.set_yticklabels([0,20,40,60,80],fontdict=font)
ax3.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax3.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax3.contourf(cyclic_lons,lat,e_ano_mean1_dec/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax3.add_patch(d)

ax3.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax3.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax3.spines['bottom'].set_visible(True)
ax3.spines['left'].set_visible(True)
ax3.spines['right'].set_visible(True)
ax3.spines['top'].set_visible(True)
ax3.spines['left'].set_linewidth(1.5)
ax3.spines['right'].set_linewidth(1.5)
ax3.spines['top'].set_linewidth(1.5)
ax3.spines['bottom'].set_linewidth(1.5)

ax3.set_title('(j) Dec',loc='left',fontsize=15,fontproperties='Arial')
ax3.set_title('300hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p1_300_dec[i,j]<0.1:
            ax3.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


ax4 = fig.add_axes([1.28, 0.7, 0.25, 0.25],projection = proj)
ax4.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax4.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax4.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax4.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax4.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax4.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax4.set_yticklabels([0,20,40,60,80],fontdict=font)
ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax4.contourf(cyclic_lons,lat,e_ano_mean1_jan/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax4.add_patch(d)

ax4.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax4.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax4.spines['bottom'].set_visible(True)
ax4.spines['left'].set_visible(True)
ax4.spines['right'].set_visible(True)
ax4.spines['top'].set_visible(True)
ax4.spines['left'].set_linewidth(1.5)
ax4.spines['right'].set_linewidth(1.5)
ax4.spines['top'].set_linewidth(1.5)
ax4.spines['bottom'].set_linewidth(1.5)

ax4.set_title('(m) Jan',loc='left',fontsize=15,fontproperties='Arial')
ax4.set_title('300hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p1_300_jan[i,j]<0.1:
            ax4.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)





ax10 = fig.add_axes([0, 0.53, 0.25, 0.25],projection = proj)
ax10.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax10.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax10.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax10.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax10.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax10.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax10.set_yticklabels([0,20,40,60,80],fontdict=font)
ax10.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax10.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax10.contourf(cyclic_lons,lat,e_ano_mean3_sep/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax10.add_patch(d)

ax10.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax10.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax10.spines['bottom'].set_visible(True)
ax10.spines['left'].set_visible(True)
ax10.spines['right'].set_visible(True)
ax10.spines['top'].set_visible(True)
ax10.spines['left'].set_linewidth(1.5)
ax10.spines['right'].set_linewidth(1.5)
ax10.spines['top'].set_linewidth(1.5)
ax10.spines['bottom'].set_linewidth(1.5)

ax10.set_title('(b) Sep',loc='left',fontsize=15,fontproperties='Arial')
ax10.set_title('850hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p3_850_sep[i,j]<0.1:
            ax10.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


ax11 = fig.add_axes([0.32, 0.53, 0.25, 0.25],projection = proj)
ax11.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax11.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax11.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax11.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax11.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax11.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax11.set_yticklabels([0,20,40,60,80],fontdict=font)
ax11.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax11.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax11.contourf(cyclic_lons,lat,e_ano_mean3_oct/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax11.add_patch(d)

ax11.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax11.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax11.spines['bottom'].set_visible(True)
ax11.spines['left'].set_visible(True)
ax11.spines['right'].set_visible(True)
ax11.spines['top'].set_visible(True)
ax11.spines['left'].set_linewidth(1.5)
ax11.spines['right'].set_linewidth(1.5)
ax11.spines['top'].set_linewidth(1.5)
ax11.spines['bottom'].set_linewidth(1.5)

ax11.set_title('(e) Oct',loc='left',fontsize=15,fontproperties='Arial')
ax11.set_title('850hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p3_850_oct[i,j]<0.1:
            ax11.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


ax12 = fig.add_axes([0.64, 0.53, 0.25, 0.25],projection = proj)
ax12.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax12.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax12.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax12.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax12.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax12.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax12.set_yticklabels([0,20,40,60,80],fontdict=font)
ax12.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax12.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax12.contourf(cyclic_lons,lat,e_ano_mean3_nov/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax12.add_patch(d)

ax12.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax12.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax12.spines['bottom'].set_visible(True)
ax12.spines['left'].set_visible(True)
ax12.spines['right'].set_visible(True)
ax12.spines['top'].set_visible(True)
ax12.spines['left'].set_linewidth(1.5)
ax12.spines['right'].set_linewidth(1.5)
ax12.spines['top'].set_linewidth(1.5)
ax12.spines['bottom'].set_linewidth(1.5)

ax12.set_title('(h) Nov',loc='left',fontsize=15,fontproperties='Arial')
ax12.set_title('850hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p3_850_nov[i,j]<0.1:
            ax12.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


ax13 = fig.add_axes([0.96, 0.53, 0.25, 0.25],projection = proj)
ax13.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax13.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax13.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax13.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax13.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax13.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax13.set_yticklabels([0,20,40,60,80],fontdict=font)
ax13.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax13.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax13.contourf(cyclic_lons,lat,e_ano_mean3_dec/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax13.add_patch(d)

ax13.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax13.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax13.spines['bottom'].set_visible(True)
ax13.spines['left'].set_visible(True)
ax13.spines['right'].set_visible(True)
ax13.spines['top'].set_visible(True)
ax13.spines['left'].set_linewidth(1.5)
ax13.spines['right'].set_linewidth(1.5)
ax13.spines['top'].set_linewidth(1.5)
ax13.spines['bottom'].set_linewidth(1.5)

ax13.set_title('(k) Dec',loc='left',fontsize=15,fontproperties='Arial')
ax13.set_title('850hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p3_850_dec[i,j]<0.1:
            ax13.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)
position=fig.add_axes([1.55,0.6,0.01,0.28])
clb=plt.colorbar(c2,cax=position,orientation='vertical',format='%.0f')
clb.ax.tick_params(labelsize=11,width=1.5)


ax14 = fig.add_axes([1.28, 0.53, 0.25, 0.25],projection = proj)
ax14.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax14.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax14.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax14.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax14.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax14.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax14.set_yticklabels([0,20,40,60,80],fontdict=font)
ax14.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax14.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax14.contourf(cyclic_lons,lat,e_ano_mean3_jan/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-70,80,10),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax14.add_patch(d)

ax14.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax14.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax14.spines['bottom'].set_visible(True)
ax14.spines['left'].set_visible(True)
ax14.spines['right'].set_visible(True)
ax14.spines['top'].set_visible(True)
ax14.spines['left'].set_linewidth(1.5)
ax14.spines['right'].set_linewidth(1.5)
ax14.spines['top'].set_linewidth(1.5)
ax14.spines['bottom'].set_linewidth(1.5)

ax14.set_title('(n) Jan',loc='left',fontsize=15,fontproperties='Arial')
ax14.set_title('850hPa HGT',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p3_850_jan[i,j]<0.1:
            ax14.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



ax5 = fig.add_axes([0, 0.36, 0.25, 0.25],projection = proj)
ax5.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax5.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax5.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax5.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax5.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax5.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax5.set_yticklabels([0,20,40,60,80],fontdict=font)
ax5.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax5.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax5.contourf(cyclic_lons,lat,e_ano_mean2_sep/100,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-5,6,1),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax5.add_patch(d)

ax5.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax5.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax5.spines['bottom'].set_visible(True)
ax5.spines['left'].set_visible(True)
ax5.spines['right'].set_visible(True)
ax5.spines['top'].set_visible(True)
ax5.spines['left'].set_linewidth(1.5)
ax5.spines['right'].set_linewidth(1.5)
ax5.spines['top'].set_linewidth(1.5)
ax5.spines['bottom'].set_linewidth(1.5)

ax5.set_title('(c) Sep',loc='left',fontsize=15,fontproperties='Arial')
ax5.set_title('SLP',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p2_500_sep[i,j]<0.1:
            ax5.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


ax6 = fig.add_axes([0.32, 0.36, 0.25, 0.25],projection = proj)
ax6.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax6.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax6.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax6.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax6.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax6.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax6.set_yticklabels([0,20,40,60,80],fontdict=font)
ax6.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax6.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax6.contourf(cyclic_lons,lat,e_ano_mean2_oct/100,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-5,6,1),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax6.add_patch(d)

ax6.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax6.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax6.spines['bottom'].set_visible(True)
ax6.spines['left'].set_visible(True)
ax6.spines['right'].set_visible(True)
ax6.spines['top'].set_visible(True)
ax6.spines['left'].set_linewidth(1.5)
ax6.spines['right'].set_linewidth(1.5)
ax6.spines['top'].set_linewidth(1.5)
ax6.spines['bottom'].set_linewidth(1.5)

ax6.set_title('(f) Oct',loc='left',fontsize=15,fontproperties='Arial')
ax6.set_title('SLP',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p2_500_oct[i,j]<0.1:
            ax6.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)




ax7 = fig.add_axes([0.64, 0.36, 0.25, 0.25],projection = proj)
ax7.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax7.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax7.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax7.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax7.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax7.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax7.set_yticklabels([0,20,40,60,80],fontdict=font)
ax7.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax7.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax7.contourf(cyclic_lons,lat,e_ano_mean2_nov/100,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-5,6,1),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax7.add_patch(d)

ax7.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax7.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax7.spines['bottom'].set_visible(True)
ax7.spines['left'].set_visible(True)
ax7.spines['right'].set_visible(True)
ax7.spines['top'].set_visible(True)
ax7.spines['left'].set_linewidth(1.5)
ax7.spines['right'].set_linewidth(1.5)
ax7.spines['top'].set_linewidth(1.5)
ax7.spines['bottom'].set_linewidth(1.5)

ax7.set_title('(i) Nov',loc='left',fontsize=15,fontproperties='Arial')
ax7.set_title('SLP',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p2_500_nov[i,j]<0.1:
            ax7.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


ax8 = fig.add_axes([0.96, 0.36, 0.25, 0.25],projection = proj)
ax8.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax8.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax8.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax8.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax8.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax8.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax8.set_yticklabels([0,20,40,60,80],fontdict=font)
ax8.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax8.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax8.contourf(cyclic_lons,lat,e_ano_mean2_dec/100,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-5,6,1),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax8.add_patch(d)

ax8.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax8.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax8.spines['bottom'].set_visible(True)
ax8.spines['left'].set_visible(True)
ax8.spines['right'].set_visible(True)
ax8.spines['top'].set_visible(True)
ax8.spines['left'].set_linewidth(1.5)
ax8.spines['right'].set_linewidth(1.5)
ax8.spines['top'].set_linewidth(1.5)
ax8.spines['bottom'].set_linewidth(1.5)

ax8.set_title('(l) Dec',loc='left',fontsize=15,fontproperties='Arial')
ax8.set_title('SLP',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p2_500_dec[i,j]<0.1:
            ax8.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



ax9 = fig.add_axes([1.28, 0.36, 0.25, 0.25],projection = proj)
ax9.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax9.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax9.set_extent([170,360,0,80],crs=ccrs.PlateCarree())
ax9.set_xticks(np.array([180,210,240,270,300,330,360]),crs=ccrs.PlateCarree())
ax9.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax9.set_xticklabels([180,210,240,270,300,330,360],fontdict=font)
ax9.set_yticklabels([0,20,40,60,80],fontdict=font)
ax9.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax9.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax9.contourf(cyclic_lons,lat,e_ano_mean2_jan/100,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-5,6,1),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax9.add_patch(d)

ax9.tick_params(axis='x',labelsize=11,width=1.5,length=7)
ax9.tick_params(axis='y',labelsize=11,width=1.5,length=7)
ax9.spines['bottom'].set_visible(True)
ax9.spines['left'].set_visible(True)
ax9.spines['right'].set_visible(True)
ax9.spines['top'].set_visible(True)
ax9.spines['left'].set_linewidth(1.5)
ax9.spines['right'].set_linewidth(1.5)
ax9.spines['top'].set_linewidth(1.5)
ax9.spines['bottom'].set_linewidth(1.5)

ax9.set_title('(o) Jan',loc='left',fontsize=15,fontproperties='Arial')
ax9.set_title('SLP',loc='right',fontsize=15,fontproperties='Arial')
for i in range(0,321,15):
    for j in range(0,1440,15):
        if p2_500_jan[i,j]<0.1:
            ax9.plot(lon[j]-265,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



position=fig.add_axes([1.55,0.42,0.01,0.13])
clb=plt.colorbar(c2,cax=position,orientation='vertical',format='%.0f')
clb.ax.tick_params(labelsize=11,width=1.5)


#plt.savefig('D:/figure/改文章6/300 500 850hpa位势高度.png',bbox_inches='tight',dpi=500)


































































































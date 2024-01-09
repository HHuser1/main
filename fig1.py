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

def sp(str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,str11,str12,str13,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13):
    pre_cli=np.array(f2.z.loc['1979-01-01':'2021-12-01',500,80:0,120:270]).reshape(43,12,321,601).mean((0))
    e1=np.array(f1.z.loc[str1,500,80:0,120:270])
    e2=np.array(f1.z.loc[str2,500,80:0,120:270])
    e3=np.array(f2.z.loc[str3,500,80:0,120:270])
    e4=np.array(f2.z.loc[str4,500,80:0,120:270])
    e5=np.array(f2.z.loc[str5,500,80:0,120:270])
    e6=np.array(f2.z.loc[str6,500,80:0,120:270])
    e7=np.array(f2.z.loc[str7,500,80:0,120:270])
    e8=np.array(f2.z.loc[str8,500,80:0,120:270])
    e9=np.array(f2.z.loc[str9,500,80:0,120:270])
    e10=np.array(f2.z.loc[str10,500,80:0,120:270])
    e11=np.array(f2.z.loc[str11,500,80:0,120:270])
    e12=np.array(f2.z.loc[str12,500,80:0,120:270])
    e13=np.array(f2.z.loc[str13,500,80:0,120:270])
    
    e_act=(e1+e2+e3+e4+e5+e6+e7+e8+e9+e10+e11+e12+e13)/13
    

    e1_ano=(e1-pre_cli[n1,:,:]).reshape(1,321,601)
    e2_ano=(e2-pre_cli[n2,:,:]).reshape(1,321,601)
    e3_ano=(e3-pre_cli[n3,:,:]).reshape(1,321,601)
    e4_ano=(e4-pre_cli[n4,:,:]).reshape(1,321,601)
    e5_ano=(e5-pre_cli[n5,:,:]).reshape(1,321,601)
    e6_ano=(e6-pre_cli[n6,:,:]).reshape(1,321,601)
    e7_ano=(e7-pre_cli[n7,:,:]).reshape(1,321,601)
    e8_ano=(e8-pre_cli[n8,:,:]).reshape(1,321,601)
    e9_ano=(e9-pre_cli[n9,:,:]).reshape(1,321,601)
    e10_ano=(e10-pre_cli[n10,:,:]).reshape(1,321,601)
    e11_ano=(e11-pre_cli[n11,:,:]).reshape(1,321,601)
    e12_ano=(e12-pre_cli[n12,:,:]).reshape(1,321,601)
    e13_ano=(e13-pre_cli[n13,:,:]).reshape(1,321,601)

    e_ano=np.concatenate((e1_ano,e2_ano,e3_ano,e4_ano,e5_ano,e6_ano,e7_ano,e8_ano,e9_ano,e10_ano,e11_ano,e12_ano,e13_ano),axis=0)
    e_ano_mean=e_ano.mean((0))
    a=np.zeros((321,601))
    t,p=ttest_ind(e_ano,a,equal_var=False)
    
    
    return e_ano_mean,p,e_act

cmap=cmaps.BlueWhiteOrangeRed

lat=np.array(f1.latitude.loc[80:0])
lon=np.array(f1.longitude.loc[120:270])

font={'family':'Arial'}
fig = plt.figure(figsize=(12,12))
proj = ccrs.PlateCarree(central_longitude=195)
e_ano_mean1,p1,e_act=sp('1956-09-01','1957-09-01','1990-09-01','2013-09-01','2014-09-01','1961-09-01','1962-09-01','1985-09-01','1989-09-01','1993-09-01','2004-09-01','2019-09-01','2020-09-01',8,8,8,8,8,8,8,8,8,8,8,8,8)
ax = fig.add_axes([0, 0.9, 0.3, 0.3],projection = proj)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax.set_yticklabels([0,20,40,60,80],fontdict=font)
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax.contourf(lon,lat,e_ano_mean1/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-50,55,5),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)

ax.tick_params(axis='x',labelsize=14,width=1.5,length=7)
ax.tick_params(axis='y',labelsize=14,width=1.5,length=7)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.set_title('(a) Sep',loc='left',fontsize=18,fontproperties='Arial')
ax.set_title('500hPa HGT',loc='right',fontsize=18,fontproperties='Arial')
for i in range(0,321,12):
    for j in range(0,601,12):
        if p1[i,j]<0.1:
            ax.plot(lon[j]-195,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



e_ano_mean2,p2,e_act=sp('1956-10-01','1957-10-01','1990-10-01','2013-10-01','2014-10-01','1961-10-01','1962-10-01','1985-10-01','1989-10-01','1993-10-01','2004-10-01','2019-10-01','2020-10-01',9,9,9,9,9,9,9,9,9,9,9,9,9)
ax1 = fig.add_axes([0, 0.67, 0.3, 0.3],projection = proj)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax1.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax1.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax1.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax1.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax1.set_yticklabels([0,20,40,60,80],fontdict=font)
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax1.contourf(lon,lat,e_ano_mean2/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-50,55,5),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax1.add_patch(d)

ax1.tick_params(axis='x',labelsize=14,width=1.5,length=7)
ax1.tick_params(axis='y',labelsize=14,width=1.5,length=7)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['top'].set_visible(True)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax1.set_title('(b) Oct',loc='left',fontsize=18,fontproperties='Arial')
ax1.set_title('500hPa HGT',loc='right',fontsize=18,fontproperties='Arial')
for i in range(0,321,12):
    for j in range(0,601,12):
        if p2[i,j]<0.1:
            ax1.plot(lon[j]-195,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



e_ano_mean3,p3,e_act=sp('1956-11-01','1957-11-01','1990-11-01','2013-11-01','2014-11-01','1961-11-01','1962-11-01','1985-11-01','1989-11-01','1993-11-01','2004-11-01','2019-11-01','2020-11-01',10,10,10,10,10,10,10,10,10,10,10,10,10)
ax2 = fig.add_axes([0, 0.44, 0.3, 0.3],projection = proj)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax2.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax2.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax2.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax2.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax2.set_yticklabels([0,20,40,60,80],fontdict=font)
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax2.contourf(lon,lat,e_ano_mean3/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-50,55,5),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
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

ax2.set_title('(c) Nov',loc='left',fontsize=18,fontproperties='Arial')
ax2.set_title('500hPa HGT',loc='right',fontsize=18,fontproperties='Arial')
for i in range(0,321,12):
    for j in range(0,601,12):
        if p3[i,j]<0.1:
            ax2.plot(lon[j]-195,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



e_ano_mean4,p4,e_act=sp('1956-12-01','1957-12-01','1990-12-01','2013-12-01','2014-12-01','1961-12-01','1962-12-01','1985-12-01','1989-12-01','1993-12-01','2004-12-01','2019-12-01','2020-12-01',11,11,11,11,11,11,11,11,11,11,11,11,11)
ax3 = fig.add_axes([0.37, 0.9, 0.3, 0.3],projection = proj)
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax3.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax3.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax3.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax3.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax3.set_yticklabels([0,20,40,60,80],fontdict=font)
ax3.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax3.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax3.contourf(lon,lat,e_ano_mean4/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-50,55,5),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax3.add_patch(d)

ax3.tick_params(axis='x',labelsize=14,width=1.5,length=7)
ax3.tick_params(axis='y',labelsize=14,width=1.5,length=7)
ax3.spines['bottom'].set_visible(True)
ax3.spines['left'].set_visible(True)
ax3.spines['right'].set_visible(True)
ax3.spines['top'].set_visible(True)
ax3.spines['left'].set_linewidth(1.5)
ax3.spines['right'].set_linewidth(1.5)
ax3.spines['top'].set_linewidth(1.5)
ax3.spines['bottom'].set_linewidth(1.5)

ax3.set_title('(d) Dec',loc='left',fontsize=18,fontproperties='Arial')
ax3.set_title('500hPa HGT',loc='right',fontsize=18,fontproperties='Arial')
for i in range(0,321,12):
    for j in range(0,601,12):
        if p4[i,j]<0.1:
            ax3.plot(lon[j]-195,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



e_ano_mean5,p5,e_act=sp('1957-01-01','1958-01-01','1991-01-01','2014-01-01','2015-01-01','1962-01-01','1963-01-01','1986-01-01','1990-01-01','1994-01-01','2005-01-01','2020-01-01','2021-01-01',0,0,0,0,0,0,0,0,0,0,0,0,0)
ax4 = fig.add_axes([0.37, 0.67, 0.3, 0.3],projection = proj)
ax4.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax4.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax4.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax4.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax4.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax4.set_yticklabels([0,20,40,60,80],fontdict=font)
ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax4.contourf(lon,lat,e_ano_mean5/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-50,55,5),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax4.add_patch(d)

ax4.tick_params(axis='x',labelsize=14,width=1.5,length=7)
ax4.tick_params(axis='y',labelsize=14,width=1.5,length=7)
ax4.spines['bottom'].set_visible(True)
ax4.spines['left'].set_visible(True)
ax4.spines['right'].set_visible(True)
ax4.spines['top'].set_visible(True)
ax4.spines['left'].set_linewidth(1.5)
ax4.spines['right'].set_linewidth(1.5)
ax4.spines['top'].set_linewidth(1.5)
ax4.spines['bottom'].set_linewidth(1.5)

ax4.set_title('(e) Jan',loc='left',fontsize=18,fontproperties='Arial')
ax4.set_title('500hPa HGT',loc='right',fontsize=18,fontproperties='Arial')
for i in range(0,321,12):
    for j in range(0,601,12):
        if p5[i,j]<0.1:
            ax4.plot(lon[j]-195,lat[i],marker='o',markersize=1.4,c='k',zorder=1)



e_ano_mean6,p6,e_act=sp('1957-02-01','1958-02-01','1991-02-01','2014-02-01','2015-02-01','1962-02-01','1963-02-01','1986-02-01','1990-02-01','1994-02-01','2005-02-01','2020-02-01','2021-02-01',1,1,1,1,1,1,1,1,1,1,1,1,1)
ax5 = fig.add_axes([0.37, 0.44, 0.3, 0.3],projection = proj)
ax5.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax5.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax5.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax5.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax5.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax5.set_yticklabels([0,20,40,60,80],fontdict=font)
ax5.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax5.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax5.contourf(lon,lat,e_ano_mean6/9.8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-50,55,5),zorder=0,cmap=cmap)

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax5.add_patch(d)

ax5.tick_params(axis='x',labelsize=14,width=1.5,length=7)
ax5.tick_params(axis='y',labelsize=14,width=1.5,length=7)
ax5.spines['bottom'].set_visible(True)
ax5.spines['left'].set_visible(True)
ax5.spines['right'].set_visible(True)
ax5.spines['top'].set_visible(True)
ax5.spines['left'].set_linewidth(1.5)
ax5.spines['right'].set_linewidth(1.5)
ax5.spines['top'].set_linewidth(1.5)
ax5.spines['bottom'].set_linewidth(1.5)

ax5.set_title('(f) Feb',loc='left',fontsize=18,fontproperties='Arial')
ax5.set_title('500hPa HGT',loc='right',fontsize=18,fontproperties='Arial')
for i in range(0,321,12):
    for j in range(0,601,12):
        if p6[i,j]<0.1:
            ax5.plot(lon[j]-195,lat[i],marker='o',markersize=1.4,c='k',zorder=1)


position=fig.add_axes([0.085,0.45,0.5,0.01])
clb=plt.colorbar(c2,cax=position,orientation='horizontal',format='%.0f',ticks=[-50,-40,-30,-20,-10,0,10,20,30,40,50])
clb.ax.tick_params(labelsize=14,width=1.5)


#plt.savefig('D:/figure/改文章/500hpa hgt.png',bbox_inches='tight',dpi=500)


































































































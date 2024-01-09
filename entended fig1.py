# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:29:50 2023

@author: asus
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

f1=xr.open_dataset('D:/data/NOAA Extended Reconstructed Sea Surface Temperature (SST) V5/sst.mnmean.nc')
sst_c=np.array(f1.sst.loc['1979-01-01':'2021-12-01',80:0,120:270]).reshape(43,12,41,76).mean((0))

sst1=np.array(f1.sst.loc['1957-02-01',80:0,120:270])-sst_c[1]
sst2=np.array(f1.sst.loc['1957-11-01',80:0,120:270])-sst_c[10]
sst3=np.array(f1.sst.loc['1962-02-01',80:0,120:270])-sst_c[1]
sst4=np.array(f1.sst.loc['1963-02-01',80:0,120:270])-sst_c[1]
sst5=np.array(f1.sst.loc['1985-11-01',80:0,120:270])-sst_c[10]
sst6=np.array(f1.sst.loc['1989-12-01',80:0,120:270])-sst_c[11]
sst7=np.array(f1.sst.loc['1991-01-01',80:0,120:270])-sst_c[0]
sst8=np.array(f1.sst.loc['1993-11-01',80:0,120:270])-sst_c[10]
sst9=np.array(f1.sst.loc['2005-01-01',80:0,120:270])-sst_c[0]
sst10=np.array(f1.sst.loc['2014-01-01',80:0,120:270])-sst_c[0]
sst11=np.array(f1.sst.loc['2015-03-01',80:0,120:270])-sst_c[2]
sst12=np.array(f1.sst.loc['2019-11-01',80:0,120:270])-sst_c[10]
sst13=np.array(f1.sst.loc['2020-11-01',80:0,120:270])-sst_c[10]

cmap=cmaps.cmp_b2r
lat=np.array(f1.lat.loc[80:0])
lon=np.array(f1.lon.loc[120:270])

font={'family':'Arial'}
fig = plt.figure(figsize=(12,12))
proj = ccrs.PlateCarree(central_longitude=120)
ax = fig.add_axes([0, 0.9, 0.25, 0.25],projection = proj)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax.set_yticklabels([0,20,40,60,80],fontdict=font)
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)
c2=ax.contourf(lon,lat,sst1,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.set_title('(a) Feb 1957',loc='left',fontsize=18,fontproperties='Arial')



ax1 = fig.add_axes([0, 0.7, 0.25, 0.25],projection = proj)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax1.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax1.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax1.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax1.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax1.set_yticklabels([0,20,40,60,80],fontdict=font)
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax1.add_patch(d)
c2=ax1.contourf(lon,lat,sst2,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax1.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax1.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['top'].set_visible(True)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax1.set_title('(b) Nov 1957',loc='left',fontsize=18,fontproperties='Arial')


ax2 = fig.add_axes([0, 0.5, 0.25, 0.25],projection = proj)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax2.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax2.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax2.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax2.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax2.set_yticklabels([0,20,40,60,80],fontdict=font)
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax2.add_patch(d)
c2=ax2.contourf(lon,lat,sst3,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax2.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax2.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(True)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['right'].set_linewidth(1.5)
ax2.spines['top'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

ax2.set_title('(c) Feb 1962',loc='left',fontsize=18,fontproperties='Arial')



ax3 = fig.add_axes([0, 0.3, 0.25, 0.25],projection = proj)
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax3.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax3.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax3.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax3.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax3.set_yticklabels([0,20,40,60,80],fontdict=font)
ax3.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax3.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax3.add_patch(d)
c2=ax3.contourf(lon,lat,sst4,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax3.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax3.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax3.spines['bottom'].set_visible(True)
ax3.spines['left'].set_visible(True)
ax3.spines['right'].set_visible(True)
ax3.spines['top'].set_visible(True)
ax3.spines['left'].set_linewidth(1.5)
ax3.spines['right'].set_linewidth(1.5)
ax3.spines['top'].set_linewidth(1.5)
ax3.spines['bottom'].set_linewidth(1.5)

ax3.set_title('(d) Feb 1963',loc='left',fontsize=18,fontproperties='Arial')



ax4 = fig.add_axes([0, 0.1, 0.25, 0.25],projection = proj)
ax4.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax4.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax4.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax4.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax4.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax4.set_yticklabels([0,20,40,60,80],fontdict=font)
ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax4.add_patch(d)
c2=ax4.contourf(lon,lat,sst5,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax4.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax4.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax4.spines['bottom'].set_visible(True)
ax4.spines['left'].set_visible(True)
ax4.spines['right'].set_visible(True)
ax4.spines['top'].set_visible(True)
ax4.spines['left'].set_linewidth(1.5)
ax4.spines['right'].set_linewidth(1.5)
ax4.spines['top'].set_linewidth(1.5)
ax4.spines['bottom'].set_linewidth(1.5)

ax4.set_title('(e) Nov 1985',loc='left',fontsize=18,fontproperties='Arial')


ax5 = fig.add_axes([0.3, 0.9, 0.25, 0.25],projection = proj)
ax5.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax5.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax5.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax5.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax5.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax5.set_yticklabels([0,20,40,60,80],fontdict=font)
ax5.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax5.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax5.add_patch(d)
c2=ax5.contourf(lon,lat,sst6,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax5.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax5.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax5.spines['bottom'].set_visible(True)
ax5.spines['left'].set_visible(True)
ax5.spines['right'].set_visible(True)
ax5.spines['top'].set_visible(True)
ax5.spines['left'].set_linewidth(1.5)
ax5.spines['right'].set_linewidth(1.5)
ax5.spines['top'].set_linewidth(1.5)
ax5.spines['bottom'].set_linewidth(1.5)

ax5.set_title('(f) Dec 1989',loc='left',fontsize=18,fontproperties='Arial')




ax6 = fig.add_axes([0.3, 0.7, 0.25, 0.25],projection = proj)
ax6.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax6.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax6.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax6.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax6.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax6.set_yticklabels([0,20,40,60,80],fontdict=font)
ax6.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax6.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax6.add_patch(d)
c2=ax6.contourf(lon,lat,sst7,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax6.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax6.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax6.spines['bottom'].set_visible(True)
ax6.spines['left'].set_visible(True)
ax6.spines['right'].set_visible(True)
ax6.spines['top'].set_visible(True)
ax6.spines['left'].set_linewidth(1.5)
ax6.spines['right'].set_linewidth(1.5)
ax6.spines['top'].set_linewidth(1.5)
ax6.spines['bottom'].set_linewidth(1.5)

ax6.set_title('(g) Jan 1991',loc='left',fontsize=18,fontproperties='Arial')




ax7 = fig.add_axes([0.3, 0.5, 0.25, 0.25],projection = proj)
ax7.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax7.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax7.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax7.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax7.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax7.set_yticklabels([0,20,40,60,80],fontdict=font)
ax7.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax7.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax7.add_patch(d)
c2=ax7.contourf(lon,lat,sst8,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax7.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax7.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax7.spines['bottom'].set_visible(True)
ax7.spines['left'].set_visible(True)
ax7.spines['right'].set_visible(True)
ax7.spines['top'].set_visible(True)
ax7.spines['left'].set_linewidth(1.5)
ax7.spines['right'].set_linewidth(1.5)
ax7.spines['top'].set_linewidth(1.5)
ax7.spines['bottom'].set_linewidth(1.5)

ax7.set_title('(h) Nov 1993',loc='left',fontsize=18,fontproperties='Arial')




ax8 = fig.add_axes([0.3, 0.3, 0.25, 0.25],projection = proj)
ax8.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax8.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax8.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax8.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax8.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax8.set_yticklabels([0,20,40,60,80],fontdict=font)
ax8.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax8.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax8.add_patch(d)
c2=ax8.contourf(lon,lat,sst9,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax8.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax8.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax8.spines['bottom'].set_visible(True)
ax8.spines['left'].set_visible(True)
ax8.spines['right'].set_visible(True)
ax8.spines['top'].set_visible(True)
ax8.spines['left'].set_linewidth(1.5)
ax8.spines['right'].set_linewidth(1.5)
ax8.spines['top'].set_linewidth(1.5)
ax8.spines['bottom'].set_linewidth(1.5)

ax8.set_title('(i) Jan 2005',loc='left',fontsize=18,fontproperties='Arial')



ax9 = fig.add_axes([0.6, 0.9, 0.25, 0.25],projection = proj)
ax9.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax9.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax9.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax9.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax9.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax9.set_yticklabels([0,20,40,60,80],fontdict=font)
ax9.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax9.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax9.add_patch(d)
c2=ax9.contourf(lon,lat,sst10,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax9.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax9.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax9.spines['bottom'].set_visible(True)
ax9.spines['left'].set_visible(True)
ax9.spines['right'].set_visible(True)
ax9.spines['top'].set_visible(True)
ax9.spines['left'].set_linewidth(1.5)
ax9.spines['right'].set_linewidth(1.5)
ax9.spines['top'].set_linewidth(1.5)
ax9.spines['bottom'].set_linewidth(1.5)

ax9.set_title('(j) Jan 2014',loc='left',fontsize=18,fontproperties='Arial')

ax10 = fig.add_axes([0.6, 0.7, 0.25, 0.25],projection = proj)
ax10.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax10.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax10.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax10.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax10.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax10.set_yticklabels([0,20,40,60,80],fontdict=font)
ax10.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax10.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax10.add_patch(d)
c2=ax10.contourf(lon,lat,sst11,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax10.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax10.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax10.spines['bottom'].set_visible(True)
ax10.spines['left'].set_visible(True)
ax10.spines['right'].set_visible(True)
ax10.spines['top'].set_visible(True)
ax10.spines['left'].set_linewidth(1.5)
ax10.spines['right'].set_linewidth(1.5)
ax10.spines['top'].set_linewidth(1.5)
ax10.spines['bottom'].set_linewidth(1.5)

ax10.set_title('(k) Mar 2015',loc='left',fontsize=18,fontproperties='Arial')



ax11 = fig.add_axes([0.6, 0.5, 0.25, 0.25],projection = proj)
ax11.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax11.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax11.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax11.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax11.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax11.set_yticklabels([0,20,40,60,80],fontdict=font)
ax11.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax11.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax11.add_patch(d)
c2=ax11.contourf(lon,lat,sst12,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax11.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax11.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax11.spines['bottom'].set_visible(True)
ax11.spines['left'].set_visible(True)
ax11.spines['right'].set_visible(True)
ax11.spines['top'].set_visible(True)
ax11.spines['left'].set_linewidth(1.5)
ax11.spines['right'].set_linewidth(1.5)
ax11.spines['top'].set_linewidth(1.5)
ax11.spines['bottom'].set_linewidth(1.5)

ax11.set_title('(l) Nov 2019',loc='left',fontsize=18,fontproperties='Arial')



ax12 = fig.add_axes([0.6, 0.3, 0.25, 0.25],projection = proj)
ax12.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,color='k')#线宽lw
ax12.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax12.set_xticks(np.array([120,150,180,210,240,270]),crs=ccrs.PlateCarree())
ax12.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax12.set_xticklabels([120,150,180,210,240,270],fontdict=font)
ax12.set_yticklabels([0,20,40,60,80],fontdict=font)
ax12.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax12.yaxis.set_major_formatter(cticker.LatitudeFormatter())

d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax12.add_patch(d)
c2=ax12.contourf(lon,lat,sst13,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-2.4,2.6,0.2),zorder=0,cmap=cmap)

ax12.tick_params(axis='x',labelsize=12,width=1.5,length=6)
ax12.tick_params(axis='y',labelsize=12,width=1.5,length=6)
ax12.spines['bottom'].set_visible(True)
ax12.spines['left'].set_visible(True)
ax12.spines['right'].set_visible(True)
ax12.spines['top'].set_visible(True)
ax12.spines['left'].set_linewidth(1.5)
ax12.spines['right'].set_linewidth(1.5)
ax12.spines['top'].set_linewidth(1.5)
ax12.spines['bottom'].set_linewidth(1.5)

ax12.set_title('(m) Nov 2020',loc='left',fontsize=18,fontproperties='Arial')

position=fig.add_axes([0.315,0.22,0.5,0.015])
clb=plt.colorbar(c2,cax=position,orientation='horizontal',format='%.1f')
clb.ax.tick_params(labelsize=14,width=1.5)
labels=clb.ax.get_xticklabels()+clb.ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]

#plt.savefig('D:/figure/改文章6/blob13事件峰值ssta.png',bbox_inches='tight',dpi=300)
















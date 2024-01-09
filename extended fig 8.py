# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:12:09 2022

@author: asus
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

from scipy.stats.mstats import ttest_ind
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cmaps
from matplotlib.patches import Ellipse 
import cmaps
fig = plt.figure(figsize=(12,12))


f1=xr.open_dataset('E:/lbm/out1/medite700/linear.t42l21.nc',decode_times=False)

lon=np.array(f1.lon)
lat=np.array(f1.lat.loc[87.8638:-2])

z5,cyclic_lons=add_cyclic_point(np.array(f1.z.loc[336.0,500,87.8638:-2,:]),coord=lon)
z10,cyclic_lons=add_cyclic_point(np.array(f1.z.loc[360.0,500,87.8638:-2,:]),coord=lon)
z15,cyclic_lons=add_cyclic_point(np.array(f1.z.loc[384.0,500,87.8638:-2,:]),coord=lon)
z20,cyclic_lons=add_cyclic_point(np.array(f1.z.loc[408.0,500,87.8638:-2,:]),coord=lon)
#z25,cyclic_lons=add_cyclic_point(np.array(f1.z.loc[576.0,500,87.8638:-2,:]),coord=lon)
#z30,cyclic_lons=add_cyclic_point(np.array(f1.z.loc[696.0,500,87.8638:-2,:]),coord=lon)
font={'family':'Arial'}

proj = ccrs.PlateCarree(central_longitude=120)
ax = fig.add_axes([0.05, 0.64, 0.4, 0.4],projection = proj)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)#线宽lw
ax.add_feature(cfeature.LAND,facecolor='none')#给陆地填色
ax.set_extent([-60,300,0,80],crs=ccrs.PlateCarree())
ax.set_xticks(np.array([-60,-60,0,60,120,180,240,300]),crs=ccrs.PlateCarree())
ax.set_yticks(np.array([0,40,80]),crs=ccrs.PlateCarree())
ax.set_xticklabels([-60,-60,0,60,120,180,240,300],fontdict=font)
ax.set_yticklabels([0,40,80],fontdict=font)
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

c2=ax.contour(cyclic_lons,lat,z5,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[1,2,3],colors='r',linestyles='-',linewidths=1.5)
c3=ax.contour(cyclic_lons,lat,z5,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[-3,-2,-1],colors='blue',linestyles='-',linewidths=1.5)
ax.clabel(c2,fontsize=10)
ax.clabel(c3,fontsize=10)
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

ax.set_title('(b) Climatology:Nov',loc='left',fontsize=16,fontproperties='Arial')
ax.set_title('500hPa HGT Day15',loc='right',fontsize=16,fontproperties='Arial')
d=Ellipse(xy=(15,40),width=30,height=10,facecolor='r',linewidth=2,linestyle='-',zorder=0,edgecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)
d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)

ax1 = fig.add_axes([0.05, 0.49, 0.4, 0.4],projection = proj)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)#线宽lw
ax1.add_feature(cfeature.LAND,facecolor='none')#给陆地填色
ax1.set_extent([-60,300,0,80],crs=ccrs.PlateCarree())
ax1.set_xticks(np.array([-60,0,60,120,180,240,300]),crs=ccrs.PlateCarree())
ax1.set_yticks(np.array([0,40,80]),crs=ccrs.PlateCarree())
ax1.set_xticklabels([-60,0,60,120,180,240,300],fontdict=font)
ax1.set_yticklabels([0,40,80],fontdict=font)
ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())

c2=ax1.contour(cyclic_lons,lat,z10,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[1,2,3],colors='r',linestyles='-',linewidths=1.5)
c3=ax1.contour(cyclic_lons,lat,z10,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[-3,-2,-1],colors='blue',linestyles='-',linewidths=1.5)
ax1.clabel(c2,fontsize=10)
ax1.clabel(c3,fontsize=10)
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

ax1.set_title('(c) Climatology:Nov',loc='left',fontsize=16,fontproperties='Arial')
ax1.set_title('500hPa HGT Day16',loc='right',fontsize=16,fontproperties='Arial')
d=Ellipse(xy=(15,40),width=30,height=10,facecolor='r',linewidth=2,linestyle='-',zorder=0,edgecolor='none',transform=ccrs.PlateCarree())
ax1.add_patch(d)
d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax1.add_patch(d)


ax2 = fig.add_axes([0.51, 0.64, 0.4, 0.4],projection = proj)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)#线宽lw
ax2.add_feature(cfeature.LAND,facecolor='none')#给陆地填色
ax2.set_extent([-60,300,0,80],crs=ccrs.PlateCarree())
ax2.set_xticks(np.array([-60,0,60,120,180,240,300]),crs=ccrs.PlateCarree())
ax2.set_yticks(np.array([0,40,80]),crs=ccrs.PlateCarree())
ax2.set_xticklabels([-60,0,60,120,180,240,300],fontdict=font)
ax2.set_yticklabels([0,40,80],fontdict=font)
ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax2.yaxis.set_major_formatter(cticker.LatitudeFormatter())

c2=ax2.contour(cyclic_lons,lat,z15,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[1,2,3],colors='r',linestyles='-',linewidths=1.5)
c3=ax2.contour(cyclic_lons,lat,z15,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[-3,-2,-1],colors='blue',linestyles='-',linewidths=1.5)
ax2.clabel(c2,fontsize=10)
ax2.clabel(c3,fontsize=10)
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

ax2.set_title('(d) Climatology:Nov',loc='left',fontsize=16,fontproperties='Arial')
ax2.set_title('500hPa HGT Day17',loc='right',fontsize=16,fontproperties='Arial')
d=Ellipse(xy=(15,40),width=30,height=10,facecolor='r',linewidth=2,linestyle='-',zorder=0,edgecolor='none',transform=ccrs.PlateCarree())
ax2.add_patch(d)
d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax2.add_patch(d)


ax3 = fig.add_axes([0.51, 0.49, 0.4, 0.4],projection = proj)
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)#线宽lw
ax3.add_feature(cfeature.LAND,facecolor='none')#给陆地填色
ax3.set_extent([-60,300,0,80],crs=ccrs.PlateCarree())
ax3.set_xticks(np.array([-60,0,60,120,180,240,300]),crs=ccrs.PlateCarree())
ax3.set_yticks(np.array([0,40,80]),crs=ccrs.PlateCarree())
ax3.set_xticklabels([-60,0,60,120,180,240,300],fontdict=font)
ax3.set_yticklabels([0,40,80],fontdict=font)
ax3.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax3.yaxis.set_major_formatter(cticker.LatitudeFormatter())

c2=ax3.contour(cyclic_lons,lat,z20,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[1,2,3],colors='r',linestyles='-',linewidths=1.5)
c3=ax3.contour(cyclic_lons,lat,z20,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[-3,-2,-1],colors='blue',linestyles='-',linewidths=1.5)
ax3.clabel(c2,fontsize=10)
ax3.clabel(c3,fontsize=10)
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

ax3.set_title('(e) Climatology:Nov',loc='left',fontsize=16,fontproperties='Arial')
ax3.set_title('500hPa HGT Day18',loc='right',fontsize=16,fontproperties='Arial')
d=Ellipse(xy=(15,40),width=30,height=10,facecolor='r',linewidth=2,linestyle='-',zorder=0,edgecolor='none',transform=ccrs.PlateCarree())
ax3.add_patch(d)
d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax3.add_patch(d)

'''
ax4 = fig.add_axes([0.51, 0.49,0.4, 0.4],projection = proj)
ax4.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)#线宽lw
ax4.add_feature(cfeature.LAND,facecolor='none')#给陆地填色
ax4.set_extent([-60,300,0,80],crs=ccrs.PlateCarree())
ax4.set_xticks(np.array([-60,0,60,120,180,240,300]),crs=ccrs.PlateCarree())
ax4.set_yticks(np.array([0,40,80]),crs=ccrs.PlateCarree())
ax4.set_xticklabels([-60,0,60,120,180,240,300],fontdict=font)
ax4.set_yticklabels([0,40,80],fontdict=font)
ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())

c2=ax4.contour(cyclic_lons,lat,z25,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[1,2,3],colors='r',linestyles='-',linewidths=1.5)
c3=ax4.contour(cyclic_lons,lat,z25,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[-3,-2,-1],colors='blue',linestyles='-',linewidths=1.5)
ax4.clabel(c2,fontsize=10)
ax4.clabel(c3,fontsize=10)
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

ax4.set_title('(e) Climatology:Nov',loc='left',fontsize=16,fontproperties='Arial')
ax4.set_title('500hPa HGT Day30',loc='right',fontsize=16,fontproperties='Arial')
d=Ellipse(xy=(15,40),width=30,height=10,facecolor='r',linewidth=2,linestyle='-',zorder=0,edgecolor='none',transform=ccrs.PlateCarree())
ax4.add_patch(d)
d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax4.add_patch(d)

ax5 = fig.add_axes([0.51, 0.34, 0.4, 0.4],projection = proj)
ax5.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5)#线宽lw
ax5.add_feature(cfeature.LAND,facecolor='none')#给陆地填色
ax5.set_extent([-60,300,0,80],crs=ccrs.PlateCarree())
ax5.set_xticks(np.array([-60,0,60,120,180,240,300]),crs=ccrs.PlateCarree())
ax5.set_yticks(np.array([0,40,80]),crs=ccrs.PlateCarree())
ax5.set_xticklabels([-60,0,60,120,180,240,300],fontdict=font)
ax5.set_yticklabels([0,40,80],fontdict=font)
ax5.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax5.yaxis.set_major_formatter(cticker.LatitudeFormatter())

c2=ax5.contour(cyclic_lons,lat,z30,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[1,2,3],colors='r',linestyles='-',linewidths=1.5)
c3=ax5.contour(cyclic_lons,lat,z30,transform=ccrs.PlateCarree(),extend='both',zorder=1,levels=[-3,-2,-1],colors='blue',linestyles='-',linewidths=1.5)
ax5.clabel(c2,fontsize=10)
ax5.clabel(c3,fontsize=10)
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

ax5.set_title('(f) Climatology:Nov',loc='left',fontsize=16,fontproperties='Arial')
ax5.set_title('500hPa HGT Day36',loc='right',fontsize=16,fontproperties='Arial')
d=Ellipse(xy=(15,40),width=30,height=10,facecolor='r',linewidth=2,linestyle='-',zorder=0,edgecolor='none',transform=ccrs.PlateCarree())
ax5.add_patch(d)
d=patches.Rectangle((200,40),25,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax5.add_patch(d)
'''

f5=xr.open_dataset('E:/lbm/frc1/medite700/frc.t42l21.nc',decode_times=False)
print(f5)
k=np.array(f5.t.loc[0])
t_mean=np.zeros((20))
for i in range(0,20):
    t_mean[i]=np.sum(k[i])/28
t_mean1=t_mean[0:11]*86400

t=np.arange(0,11,1)
font={'family':'Arial'}
totalq_ano_mean=np.array([0.0313,0.0407,0.0636,0.0809,0.2735,0.2677,0.2768,0.3063,0.3850,0.3383,-0.2158])
ax6 = fig.add_axes([-0.17, 0.67, 0.16, 0.2])
#ax.set_yscale('symlog')
ax6.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
ax6.set_yticklabels(['1000','950','900','850','700','600','500','400','300','250','200'],fontdict=font)
ax6.set_xticks([0,0.4,0.8,1.2])
#ax.set_xticklabels(['0','0.4','0.8','1.2'],fontdict=font)
ax6.set_xlim(-0.4,0.6)
x_major_locator=MultipleLocator(0.2)
x_minor_locator=MultipleLocator(0.1)
ax6.xaxis.set_minor_locator(x_minor_locator)
ax6.xaxis.set_major_locator(x_major_locator)
#ax6.yaxis.tick_right()
#ax6.yaxis.set_label_position('right')
#ax.invert_yaxis()
ax6.autoscale(tight=True,axis='y')
ax6.plot(totalq_ano_mean,t,color='blue',linewidth=2,label='q1')
ax6.plot(t_mean1,t,color='red',linewidth=2,label='heating')
ax6.axvline(0,color='k',linewidth=1,linestyle='--')
ax6.set_ylabel('Pressure (hPa)',fontsize=12,fontproperties='Arial')
ax6.tick_params(labelsize=12,width=1.5,length=6)
ax6.tick_params(width=1.5,length=3,which='minor')
ax6.spines['bottom'].set_visible(True)
ax6.spines['left'].set_visible(True)
ax6.spines['right'].set_visible(True)
ax6.spines['top'].set_visible(True)
ax6.spines['left'].set_linewidth(1.5)
ax6.spines['right'].set_linewidth(1.5)
ax6.spines['top'].set_linewidth(1.5)
ax6.spines['bottom'].set_linewidth(1.5)
#ax6.set_title('(a)',loc='left',fontsize=16,fontproperties='Arial')
ax6.set_title('K/day',loc='right',fontsize=16,fontproperties='Arial')
ax6.legend(fontsize=9)
ax6.set_title('(a)',loc='left',fontsize=16,fontproperties='Arial')

plt.savefig('D:/figure改文章12-25/地中海700lbm.png',format='png',bbox_inches='tight',dpi=300)
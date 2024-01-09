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
import cmaps

f1=xr.open_dataset('D:/monthly.nc')

lat=np.array(f1.lat.loc[0:80])
lon=np.array(f1.lon.loc[:])
lev=np.array(f1.lev.loc[1000:200])
lev1=lev.reshape(10,1,1)
dlat=(np.gradient(lat)*np.pi/180.0).reshape(81,1)
dlon=(np.gradient(lon)*np.pi/180.0).reshape(1,360)
coslat=(np.cos(np.array(lat)*np.pi/180.0)).reshape(81,1)
a=6371000
dx=a*coslat*dlon
dy=(a*dlat)
dp=np.gradient(lev1,axis=0)


u=np.array(f1.u.loc['1950-01-01':'2021-12-01',1000:200,0:80,:])
v=np.array(f1.v.loc['1950-01-01':'2021-12-01',1000:200,0:80,:])
w=np.array(f1.w.loc['1950-01-01':'2021-12-01',1000:200,0:80,:])
t=np.array(f1.t.loc['1950-01-01':'2021-12-01',1000:200,0:80,:])
term1=np.zeros((864,10,81,360))
for i in range(0,864,1):
    if i==0:
        term1[i,:,:,:]=t[i+1,:,:,:]-t[i,:,:,:]
    if i>0:
        term1[i,:,:,:]=t[i,:,:,:]-t[i-1,:,:,:]
    

term2=w*(((287*t)/(1004*lev1))-(np.gradient(t,axis=1)/np.gradient(lev1,axis=0)))
term3=u*(np.gradient(t,axis=3)/dx)+v*(np.gradient(t,axis=2)/dy)
q=(term1/30)-(24*36*term2)+(24*3600*term3)


q_c=q.reshape(72,12,10,81,360).mean((0))
q_c_medi=q_c.mean((2,3))

q_c=(q.reshape(72,12,10,81,360))[28:72,:,:,:,:].mean((0))

q_ano1=(q[82,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano2=(q[94,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano3=(q[490,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano4=(q[766,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano5=(q[778,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano6=(q[142,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano7=(q[154,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano8=(q[430,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano9=(q[478,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano10=(q[526,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano11=(q[658,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano12=(q[838,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)
q_ano13=(q[850,7,:,:]-q_c[10,7,:,:]).reshape(1,81,360)

q_ano_nov=np.concatenate((q_ano1,q_ano2,q_ano3,q_ano4,q_ano5,q_ano6,q_ano7,q_ano8,q_ano9,q_ano10,q_ano11,q_ano12,q_ano13),axis=0)
q_ano_nov_mean=q_ano_nov.mean((0))

qq_ano_nov_mean=(q_ano_nov[:,35:45,0:30]).mean((0,1,2))

a=np.zeros((81,360))
t,p=ttest_ind(q_ano_nov,a,equal_var=False)


cyclic_data,cyclic_lons=add_cyclic_point(q_ano_nov_mean,coord=lon)

font={'family':'Arial'}
fig = plt.figure(figsize=(12,12))
proj = ccrs.PlateCarree(central_longitude=60)  

cmap=cmaps.ViBlGrWhYeOrRe    
ax = fig.add_axes([0.3, 0.4, 0.6, 0.6],projection = proj)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'),lw=0.5)#线宽lw
ax.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax.set_extent([-30,150,0,80],crs=ccrs.PlateCarree())
ax.set_xticks(np.array(np.arange(-30,180,30)),crs=ccrs.PlateCarree())
ax.set_yticks(np.array([0,20,40,60,80]),crs=ccrs.PlateCarree())
ax.set_xticklabels(np.arange(-30,180,30),fontdict=font)
ax.set_yticklabels([0,20,40,60,80],fontdict=font)
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax.contourf(cyclic_lons,lat,cyclic_data,transform=ccrs.PlateCarree(),extend='both',levels=np.arange(-0.8,1.0,0.2),zorder=0,cmap=cmap)
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

#ax.set_title('(a) Nov',loc='left',fontsize=18,fontproperties='Arial')
#ax.set_title('300hPa q1',loc='right',fontsize=18,fontproperties='Arial')
d=patches.Rectangle((0,35),30,10,linewidth=2,linestyle='-',zorder=4,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)

d=patches.Rectangle((334,48),26,17,linewidth=2,linestyle='-',zorder=4,edgecolor='k',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)

for i in range(0,81):
    for j in range(0,360):
        if p[i,j]<0.1 and 0<=j<60:
            ax.plot(lon[j]-60,lat[i],marker='o',markersize=1,c='white',zorder=3)
        if p[i,j]<0.1 and 60<=j<240:
            ax.plot(lon[j]-60,lat[i],marker='o',markersize=1,c='white',zorder=3)
        if p[i,j]<0.1 and 240<=j<=360:
            ax.plot(lon[j]-420,lat[i],marker='o',markersize=1,c='white',zorder=3) 

position=fig.add_axes([0.91,0.565,0.015,0.27])
clb=plt.colorbar(c2,cax=position,orientation='vertical',format='%.1f')
clb.ax.tick_params(labelsize=14,width=1.5)


plt.savefig('D:/figure/改文章6/11月300hPa热源.pdf',dpi=300,bbox_inches='tight')  
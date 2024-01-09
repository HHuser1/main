# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:47:07 2023

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
from scipy import stats
import cmaps
from matplotlib.colors import ListedColormap

f1=xr.open_dataset('D:/data/JRA55/velocity potential 2.5/vp 1958-2021.nc')
f2=xr.open_dataset('D:/data/JRA55/relative vorticity/relative1958-2021.nc')
f3=xr.open_dataset('D:/data/JRA55/relative divergence/relative divergence1958-2021.nc')


vp_c=np.array(f1.vp.loc['1979-01-01':'2021-12-01',300,90:0,:]).reshape(43,12,37,144).mean((0))
vor_c=np.array(f2.vo.loc['1979-01-01':'2021-12-01',300,90:0,:]).reshape(43,12,37,144).mean((0))

lat=np.array(f1.latitude.loc[90:0])
lon=np.array(f1.longitude)
dlat=(np.gradient(lat)*np.pi/180.0).reshape(37,1)
dlon=(np.gradient(lon)*np.pi/180.0).reshape(1,144)
coslat=(np.cos(np.array(lat)*np.pi/180.0)).reshape(37,1)
a=6371000
dx=a*coslat*dlon
dy=(a*dlat)*-1
lev=np.array(f1.isobaricInhPa)

f8=2*7.292e-5*np.sin(lat*np.pi/180.0)
f=np.zeros((37,144))

for i in range(0,37):
    f[i,:]=f8[i]
def rws(str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,str11,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11):
    
    u_c=np.gradient(vp_c,axis=2)/dx
    v_c=np.gradient(vp_c,axis=1)/dy
    
    vp1=np.array(f1.vp.loc[str1,300,90:0,:])
    vor1=np.array(f2.vo.loc[str1,300,90:0,:])
    u_div1=np.gradient(vp1,axis=1)/dx
    v_div1=np.gradient(vp1,axis=0)/dy
    u_ano1=u_div1-u_c[n1]
    v_ano1=v_div1-v_c[n1]
    vor_ano1=np.array(f2.vo.loc[str1,300,90:0,:])-vor_c[n1]
    term1_1=np.gradient(u_ano1*(f+vor_c[n1]),axis=1)/dx
    term2_1=np.gradient(v_ano1*(f+vor_c[n1]),axis=0)/dy
    term3_1=np.gradient(u_c[n1]*vor_ano1,axis=1)/dx
    term4_1=np.gradient(v_c[n1]*vor_ano1,axis=0)/dy
    term5_1=np.gradient(u_ano1*vor_ano1,axis=1)/dx
    term6_1=np.gradient(v_ano1*vor_ano1,axis=0)/dy
    equa_1=-(term1_1+term2_1+term3_1+term4_1+term5_1+term6_1)*10e10

    vp2=np.array(f1.vp.loc[str2,300,90:0,:])
    vor2=np.array(f2.vo.loc[str2,300,90:0,:])
    u_div2=np.gradient(vp2,axis=1)/dx
    v_div2=np.gradient(vp2,axis=0)/dy
    u_ano2=u_div2-u_c[n2]
    v_ano2=v_div2-v_c[n2]
    vor_ano2=np.array(f2.vo.loc[str2,300,90:0,:])-vor_c[n2]
    term1_2=np.gradient(u_ano2*(f+vor_c[n2]),axis=1)/dx
    term2_2=np.gradient(v_ano2*(f+vor_c[n2]),axis=0)/dy
    term3_2=np.gradient(u_c[n2]*vor_ano2,axis=1)/dx
    term4_2=np.gradient(v_c[n2]*vor_ano2,axis=0)/dy
    term5_2=np.gradient(u_ano2*vor_ano2,axis=1)/dx
    term6_2=np.gradient(v_ano2*vor_ano2,axis=0)/dy
    equa_2=-(term1_2+term2_2+term3_2+term4_2+term5_2+term6_2)*10e10
    
    vp3=np.array(f1.vp.loc[str3,300,90:0,:])
    vor3=np.array(f2.vo.loc[str3,300,90:0,:])
    u_div3=np.gradient(vp3,axis=1)/dx
    v_div3=np.gradient(vp3,axis=0)/dy
    u_ano3=u_div3-u_c[n3]
    v_ano3=v_div3-v_c[n3]
    vor_ano3=np.array(f2.vo.loc[str3,300,90:0,:])-vor_c[n3]
    term1_3=np.gradient(u_ano3*(f+vor_c[n3]),axis=1)/dx
    term2_3=np.gradient(v_ano3*(f+vor_c[n3]),axis=0)/dy
    term3_3=np.gradient(u_c[n3]*vor_ano3,axis=1)/dx
    term4_3=np.gradient(v_c[n3]*vor_ano3,axis=0)/dy
    term5_3=np.gradient(u_ano3*vor_ano3,axis=1)/dx
    term6_3=np.gradient(v_ano3*vor_ano3,axis=0)/dy
    equa_3=-(term1_3+term2_3+term3_3+term4_3+term5_3+term6_3)*10e10
    
    vp4=np.array(f1.vp.loc[str4,300,90:0,:])
    vor4=np.array(f2.vo.loc[str4,300,90:0,:])
    u_div4=np.gradient(vp4,axis=1)/dx
    v_div4=np.gradient(vp4,axis=0)/dy
    u_ano4=u_div4-u_c[n4]
    v_ano4=v_div4-v_c[n4]
    vor_ano4=np.array(f2.vo.loc[str4,300,90:0,:])-vor_c[n4]
    term1_4=np.gradient(u_ano4*(f+vor_c[n4]),axis=1)/dx
    term2_4=np.gradient(v_ano4*(f+vor_c[n4]),axis=0)/dy
    term3_4=np.gradient(u_c[n4]*vor_ano4,axis=1)/dx
    term4_4=np.gradient(v_c[n4]*vor_ano4,axis=0)/dy
    term5_4=np.gradient(u_ano4*vor_ano4,axis=1)/dx
    term6_4=np.gradient(v_ano4*vor_ano4,axis=0)/dy
    equa_4=-(term1_4+term2_4+term3_4+term4_4+term5_4+term6_4)*10e10
    
    vp5=np.array(f1.vp.loc[str5,300,90:0,:])
    vor5=np.array(f2.vo.loc[str5,300,90:0,:])
    u_div5=np.gradient(vp5,axis=1)/dx
    v_div5=np.gradient(vp5,axis=0)/dy
    u_ano5=u_div5-u_c[n5]
    v_ano5=v_div5-v_c[n5]
    vor_ano5=np.array(f2.vo.loc[str5,300,90:0,:])-vor_c[n5]
    term1_5=np.gradient(u_ano5*(f+vor_c[n5]),axis=1)/dx
    term2_5=np.gradient(v_ano5*(f+vor_c[n5]),axis=0)/dy
    term3_5=np.gradient(u_c[n5]*vor_ano5,axis=1)/dx
    term4_5=np.gradient(v_c[n5]*vor_ano5,axis=0)/dy
    term5_5=np.gradient(u_ano5*vor_ano5,axis=1)/dx
    term6_5=np.gradient(v_ano5*vor_ano5,axis=0)/dy
    equa_5=-(term1_5+term2_5+term3_5+term4_5+term5_5+term6_5)*10e10
    
    vp6=np.array(f1.vp.loc[str6,300,90:0,:])
    vor6=np.array(f2.vo.loc[str6,300,90:0,:])
    u_div6=np.gradient(vp6,axis=1)/dx
    v_div6=np.gradient(vp6,axis=0)/dy
    u_ano6=u_div6-u_c[n6]
    v_ano6=v_div6-v_c[n6]
    vor_ano6=np.array(f2.vo.loc[str6,300,90:0,:])-vor_c[n6]
    term1_6=np.gradient(u_ano6*(f+vor_c[n6]),axis=1)/dx
    term2_6=np.gradient(v_ano6*(f+vor_c[n6]),axis=0)/dy
    term3_6=np.gradient(u_c[n6]*vor_ano6,axis=1)/dx
    term4_6=np.gradient(v_c[n6]*vor_ano6,axis=0)/dy
    term5_6=np.gradient(u_ano6*vor_ano6,axis=1)/dx
    term6_6=np.gradient(v_ano6*vor_ano6,axis=0)/dy
    equa_6=-(term1_6+term2_6+term3_6+term4_6+term5_6+term6_6)*10e10
    
    vp7=np.array(f1.vp.loc[str7,300,90:0,:])
    vor7=np.array(f2.vo.loc[str7,300,90:0,:])
    u_div7=np.gradient(vp7,axis=1)/dx
    v_div7=np.gradient(vp7,axis=0)/dy
    u_ano7=u_div7-u_c[n7]
    v_ano7=v_div7-v_c[n7]
    vor_ano7=np.array(f2.vo.loc[str7,300,90:0,:])-vor_c[n7]
    term1_7=np.gradient(u_ano7*(f+vor_c[n7]),axis=1)/dx
    term2_7=np.gradient(v_ano7*(f+vor_c[n7]),axis=0)/dy
    term3_7=np.gradient(u_c[n7]*vor_ano7,axis=1)/dx
    term4_7=np.gradient(v_c[n7]*vor_ano7,axis=0)/dy
    term5_7=np.gradient(u_ano7*vor_ano7,axis=1)/dx
    term6_7=np.gradient(v_ano7*vor_ano7,axis=0)/dy
    equa_7=-(term1_7+term2_7+term3_7+term4_7+term5_7+term6_7)*10e10
    
    vp8=np.array(f1.vp.loc[str8,300,90:0,:])
    vor8=np.array(f2.vo.loc[str8,300,90:0,:])
    u_div8=np.gradient(vp8,axis=1)/dx
    v_div8=np.gradient(vp8,axis=0)/dy
    u_ano8=u_div8-u_c[n8]
    v_ano8=v_div8-v_c[n8]
    vor_ano8=np.array(f2.vo.loc[str8,300,90:0,:])-vor_c[n8]
    term1_8=np.gradient(u_ano8*(f+vor_c[n8]),axis=1)/dx
    term2_8=np.gradient(v_ano8*(f+vor_c[n8]),axis=0)/dy
    term3_8=np.gradient(u_c[n8]*vor_ano8,axis=1)/dx
    term4_8=np.gradient(v_c[n8]*vor_ano8,axis=0)/dy
    term5_8=np.gradient(u_ano8*vor_ano8,axis=1)/dx
    term6_8=np.gradient(v_ano8*vor_ano8,axis=0)/dy
    equa_8=-(term1_8+term2_8+term3_8+term4_8+term5_8+term6_8)*10e10
    
    vp9=np.array(f1.vp.loc[str9,300,90:0,:])
    vor9=np.array(f2.vo.loc[str9,300,90:0,:])
    u_div9=np.gradient(vp9,axis=1)/dx
    v_div9=np.gradient(vp9,axis=0)/dy
    u_ano9=u_div9-u_c[n9]
    v_ano9=v_div9-v_c[n9]
    vor_ano9=np.array(f2.vo.loc[str9,300,90:0,:])-vor_c[n9]
    term1_9=np.gradient(u_ano9*(f+vor_c[n9]),axis=1)/dx
    term2_9=np.gradient(v_ano9*(f+vor_c[n9]),axis=0)/dy
    term3_9=np.gradient(u_c[n9]*vor_ano9,axis=1)/dx
    term4_9=np.gradient(v_c[n9]*vor_ano9,axis=0)/dy
    term5_9=np.gradient(u_ano9*vor_ano9,axis=1)/dx
    term6_9=np.gradient(v_ano9*vor_ano9,axis=0)/dy
    equa_9=-(term1_9+term2_9+term3_9+term4_9+term5_9+term6_9)*10e10
    
    vp10=np.array(f1.vp.loc[str10,300,90:0,:])
    vor10=np.array(f2.vo.loc[str10,300,90:0,:])
    u_div10=np.gradient(vp10,axis=1)/dx
    v_div10=np.gradient(vp10,axis=0)/dy
    u_ano10=u_div10-u_c[n10]
    v_ano10=v_div10-v_c[n10]
    vor_ano10=np.array(f2.vo.loc[str10,300,90:0,:])-vor_c[n10]
    term1_10=np.gradient(u_ano10*(f+vor_c[n10]),axis=1)/dx
    term2_10=np.gradient(v_ano10*(f+vor_c[n10]),axis=0)/dy
    term3_10=np.gradient(u_c[n10]*vor_ano10,axis=1)/dx
    term4_10=np.gradient(v_c[n10]*vor_ano10,axis=0)/dy
    term5_10=np.gradient(u_ano10*vor_ano10,axis=1)/dx
    term6_10=np.gradient(v_ano10*vor_ano10,axis=0)/dy
    equa_10=-(term1_10+term2_10+term3_10+term4_10+term5_10+term6_10)*10e10
    
    vp11=np.array(f1.vp.loc[str11,300,90:0,:])
    vor11=np.array(f2.vo.loc[str11,300,90:0,:])
    u_div11=np.gradient(vp11,axis=1)/dx
    v_div11=np.gradient(vp11,axis=0)/dy
    u_ano11=u_div11-u_c[n11]
    v_ano11=v_div11-v_c[n11]
    vor_ano11=np.array(f2.vo.loc[str11,300,90:0,:])-vor_c[n11]
    term1_11=np.gradient(u_ano11*(f+vor_c[n11]),axis=1)/dx
    term2_11=np.gradient(v_ano11*(f+vor_c[n11]),axis=0)/dy
    term3_11=np.gradient(u_c[n11]*vor_ano11,axis=1)/dx
    term4_11=np.gradient(v_c[n11]*vor_ano11,axis=0)/dy
    term5_11=np.gradient(u_ano11*vor_ano11,axis=1)/dx
    term6_11=np.gradient(v_ano11*vor_ano11,axis=0)/dy
    equa_11=-(term1_11+term2_11+term3_11+term4_11+term5_11+term6_11)*10e10
    
    equa=(equa_1+equa_2+equa_3+equa_4+equa_5+equa_6+equa_7+equa_8+equa_9+equa_10+equa_11)/11
    
    equ1=equa_1.reshape(1,37,144)
    equ2=equa_2.reshape(1,37,144)
    equ3=equa_3.reshape(1,37,144)
    equ4=equa_4.reshape(1,37,144)
    equ5=equa_5.reshape(1,37,144)
    equ6=equa_6.reshape(1,37,144)
    equ7=equa_7.reshape(1,37,144)
    equ8=equa_8.reshape(1,37,144)
    equ9=equa_9.reshape(1,37,144)
    equ10=equa_10.reshape(1,37,144)
    equ11=equa_11.reshape(1,37,144)
    equ=np.concatenate((equ1,equ2,equ3,equ4,equ5,equ6,equ7,equ8,equ9,equ10,equ11),axis=0)
    a=np.zeros((37,144))
    t,p=ttest_ind(equ,a,equal_var=False)
    return equa,p

div_c=np.array(f3.d.loc['1979-01-01':'2021-12-01',200,90:0,:]).reshape(43,12,37,144).mean((0))
def div(str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,str11,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11):
    div_ano1=(np.array(f3.d.loc[str1,200,90:0,:])-div_c[n1]).reshape(1,37,144)
    div_ano2=(np.array(f3.d.loc[str2,200,90:0,:])-div_c[n2]).reshape(1,37,144)
    div_ano3=(np.array(f3.d.loc[str3,200,90:0,:])-div_c[n3]).reshape(1,37,144)
    div_ano4=(np.array(f3.d.loc[str4,200,90:0,:])-div_c[n4]).reshape(1,37,144)
    div_ano5=(np.array(f3.d.loc[str5,200,90:0,:])-div_c[n5]).reshape(1,37,144)
    div_ano6=(np.array(f3.d.loc[str6,200,90:0,:])-div_c[n6]).reshape(1,37,144)
    div_ano7=(np.array(f3.d.loc[str7,200,90:0,:])-div_c[n7]).reshape(1,37,144)
    div_ano8=(np.array(f3.d.loc[str8,200,90:0,:])-div_c[n8]).reshape(1,37,144)
    div_ano9=(np.array(f3.d.loc[str9,200,90:0,:])-div_c[n9]).reshape(1,37,144)
    div_ano10=(np.array(f3.d.loc[str10,200,90:0,:])-div_c[n10]).reshape(1,37,144)
    div_ano11=(np.array(f3.d.loc[str11,200,90:0,:])-div_c[n11]).reshape(1,37,144)
    
    div_ano=np.concatenate((div_ano1,div_ano2,div_ano3,div_ano4,div_ano5,div_ano6,div_ano7,div_ano8,div_ano9,div_ano10,div_ano11),axis=0)
    div_ano_mean=div_ano.mean((0))
    a=np.zeros((37,144))
    t,p=ttest_ind(div_ano,a,equal_var=False)
    return div_ano_mean,p

cmap=cmaps.MPL_coolwarm
newcolors=cmap(np.linspace(0, 1, 16))
newcolors1=np.array([newcolors[0,:],newcolors[1,:],newcolors[2,:],newcolors[3,:],newcolors[4,:],newcolors[5,:],newcolors[6,:],[1,1,1,1],[1,1,1,1],newcolors[9,:],newcolors[10,:],newcolors[11,:],newcolors[12,:],newcolors[13,:],newcolors[14,:],newcolors[15,:]])
newcolors2=tuple(newcolors1)
newcmap1=ListedColormap(newcolors2)


font={'family':'Arial'}
equa,p=rws('1990-11-01','2013-11-01','2014-11-01','1961-11-01','1962-11-01','1985-11-01','1989-11-01','1993-11-01','2004-11-01','2019-11-01','2020-11-01',10,10,10,10,10,10,10,10,10,10,10)

cyclic_data,cyclic_lons=add_cyclic_point(equa,coord=lon)
cyclic_p,cyclic_lons=add_cyclic_point(p,coord=lon)
cyclic_data1=cyclic_data[1:4,:]/50
cyclic_data[1:4,:]=cyclic_data1
fig = plt.figure(figsize=(12,12))
proj = ccrs.PlateCarree(central_longitude=120)  


ax = fig.add_axes([0.3, 0.39, 0.6, 0.6],projection = proj)

y_major_locator=MultipleLocator(20)
y_minor_locator=MultipleLocator(10)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
x_major_locator=MultipleLocator(60)
x_minor_locator=MultipleLocator(10)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.xaxis.set_major_locator(x_major_locator)

ax.add_feature(cfeature.COASTLINE.with_scale('10m'),lw=0.5)#线宽lw
ax.add_feature(cfeature.LAND,facecolor='gray')#给陆地填色
ax.set_extent([-30,150,0,90],crs=ccrs.PlateCarree())
ax.set_xticks(np.array(np.arange(-30,180,30)),crs=ccrs.PlateCarree())
ax.set_yticks(np.array([20,40,60,80]),crs=ccrs.PlateCarree())
ax.set_xticklabels(np.arange(-30,180,30),fontdict=font)
ax.set_yticklabels([20,40,60,80],fontdict=font)
ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
c2=ax.contourf(cyclic_lons,lat,cyclic_data,transform=ccrs.PlateCarree(),levels=np.arange(-8,9,1),zorder=0,extend='both',cmap=newcmap1)
ax.tick_params(labelsize=12,width=0.5,length=6,which='major',top='True',right='True',pad=0.5)
ax.tick_params(width=0.5,length=4,which='minor',top='True',right='True')
'''
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
'''
#ax.set_title('(a) Nov',loc='left',fontsize=13,fontproperties='Times New Roman')
#ax.set_title('Nov 300hPa RWS',loc='right',fontsize=12,fontproperties='Times New Roman',pad=5)
d=patches.Rectangle((0,35),30,10,linewidth=2,linestyle='-',zorder=5,edgecolor='lime',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)

d=patches.Rectangle((334,48),26,17,linewidth=2,linestyle='-',zorder=5,edgecolor='k',facecolor='none',transform=ccrs.PlateCarree())
ax.add_patch(d)
ax.set_aspect(0.8)
'''
div,p=div('1990-11-01','2013-11-01','2014-11-01','1961-11-01','1962-11-01','1985-11-01','1989-11-01','1993-11-01','2004-11-01','2019-11-01','2020-11-01',10,10,10,10,10,10,10,10,10,10,10)
div_p=np.zeros((37,144))
for i in range(0,37):
    for j in range(0,144):
        if p[i,j]<0.1:
            div_p[i,j]=div[i,j]
        else:
            div_p[i,j]=np.nan
           
clp=ax.contourf(lon,lat,div_p*1e7,levels=[0,100,100000],hatches=['+++',None],colors='none',zorder=4,transform=ccrs.PlateCarree())
clp1=ax.contourf(lon,lat,div_p*1e7,levels=[-100,0,100000],hatches=['+++',None],colors='none',zorder=4,transform=ccrs.PlateCarree())



for collection in clp.collections:
    collection.set_edgecolor('fuchsia')
for collection in clp.collections:
    collection.set_linewidth(0)
    
for collection in clp1.collections:
    collection.set_edgecolor('limegreen')
for collection in clp1.collections:
    collection.set_linewidth(0)

   

for i in range(0,37):
    for j in range(0,144):
        if div_p[i,j]>0 and 0<=j<48:
            ax.plot(lon[j]-120,lat[i],marker='*',markersize=4,c='fuchsia',zorder=3)
        if div_p[i,j]>0 and 48<=j<120:
            ax.plot(lon[j]-120,lat[i],marker='*',markersize=4,c='fuchsia',zorder=3)
        if div_p[i,j]>0 and 120<=j<=143:
            ax.plot(lon[j]-480,lat[i],marker='*',markersize=4,c='fuchsia',zorder=3)  
            
for i in range(0,37):
    for j in range(0,144):
        if div_p[i,j]<0 and 0<=j<48:
            ax.plot(lon[j]-120,lat[i],marker='*',markersize=4,c='limegreen',zorder=3)
        if div_p[i,j]<0 and 48<=j<120:
            ax.plot(lon[j]-120,lat[i],marker='*',markersize=4,c='limegreen',zorder=3)
        if div_p[i,j]<0 and 120<=j<=143:
            ax.plot(lon[j]-480,lat[i],marker='*',markersize=4,c='limegreen',zorder=3) 
'''    
for i in range(1,36):
    for j in range(0,144):
        if cyclic_p[i,j]<0.1 and 0<=j<48:
            ax.plot(lon[j]-120,lat[i],marker='o',markersize=1.5,c='white',zorder=4)
        if cyclic_p[i,j]<0.1 and 48<=j<120:
            ax.plot(lon[j]-120,lat[i],marker='o',markersize=1.5,c='white',zorder=4)
        if cyclic_p[i,j]<0.1 and 120<=j<=143:
            ax.plot(lon[j]-480,lat[i],marker='o',markersize=1.5,c='white',zorder=4)        
position=fig.add_axes([0.91,0.57,0.013,0.24])
clb=plt.colorbar(c2,cax=position,orientation='vertical',format='%.0f')
clb.ax.tick_params(labelsize=11,width=0.7)
clb.set_ticks([-8,-4,0,4,8])
clb.set_ticklabels(('-8','-4','0','4','8'))
#plt.savefig('D:/figure改文章12-25/地中海11月300hParws.pdf',format='pdf',dpi=300,bbox_inches='tight')  
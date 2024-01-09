;==============================================================
;======================= test1 +test2 ---------v2
;==============================================================
tx = 0.1
nday1 = 15
nday2 = 17

;-------------------------data 1 
f0 = addfile("/mnt/d/shi/code1/data/1/ncepnov.t42l21.nc", "r")
uc300 = f0->u(0,{300},:,:)
f1 = addfile("/mnt/d/shi/code1/data/1/linear.t42l21.nc", "r")
z3001 = f1->z(nday1:nday2,{300},:,:)
u3001 = f1->u(nday1:nday2,{300},:,:)
;-------------------------------plot 1 经向风异常300百帕
v3001 = f1->v(nday1:nday2,{300},:,:)
;-------------------------------plot 2 500百帕hgt & uv
z5001 = f1->z(nday1:nday2,{500},:,:)
u5001 = f1->u(nday1:nday2,{500},:,:)
v5001 = f1->v(nday1:nday2,{500},:,:)
;-------------------------------plot 3 850百帕hgt & uv
z8501 = f1->z(nday1:nday2,{850},:,:)
u8501 = f1->u(nday1:nday2,{850},:,:)
v8501 = f1->v(nday1:nday2,{850},:,:)
;-------------------------------plot 4 SLP &uv
z10001 = f1->p(nday1:nday2,{1000},:,:)
u10001 = f1->u(nday1:nday2,{1000},:,:)
v10001 = f1->v(nday1:nday2,{1000},:,:)
; printVarSummary(z500)
; exit
;---------------------avg 
z300 = dim_avg_n_Wrap(z3001, 0)
u300 = dim_avg_n_Wrap(u3001, 0)
v300 = dim_avg_n_Wrap(v3001, 0)
z500 = dim_avg_n_Wrap(z5001, 0)
u500 = dim_avg_n_Wrap(u5001, 0)
v500 = dim_avg_n_Wrap(v5001, 0)
z850 = dim_avg_n_Wrap(z8501, 0)
u850 = dim_avg_n_Wrap(u8501, 0)
v850 = dim_avg_n_Wrap(v8501, 0)
z1000 = dim_avg_n_Wrap(z10001, 0)
u1000 = dim_avg_n_Wrap(u10001, 0)
v1000 = dim_avg_n_Wrap(v10001, 0)

f2 = addfile("/mnt/d/shi/code1/case2-lbm-data-v2-tast1.nc","r")
Fx = f2->Fx1
Fy = f2->Fy1
f3 = addfile("/mnt/d/shi/code1/data/1/v1/frc.t42l21.nc","r")
tt1 = f3->t(0,:,:,:)
tt = tt1*86400;100000
copy_VarMeta(tt1, tt)

tp1 = tt({1000:200},{45:35},{0:30})
sum1 = new((/11/),"float")
num1 = new((/11/),"float")
t_profile = new((/11/),"float")
sum1 = 0
num1 = 0
do i = 0,10
    do j = 0,2
        do k = 0,10
            if(tp1(i,j,k).eq.0) then 
               sum1(i) = sum1(i) 
               num1(i) = num1(i) 
           else 
               sum1(i) = sum1(i) + tp1(i,j,k)
               num1(i) = num1(i) + 1
           end if 
       end do 
   end do 
end do 
do i = 0,10
   t_profile(i) = sum1(i)/28
end do
copy_VarMeta(tt(0:10,0,0), t_profile)


f4 = "/mnt/d/shi/code1/data/1/q1.txt"
y1 = asciiread(f4, (/11/), "float")
copy_VarMeta(tt(0:10,0,0), y1)
;-------------------------data 2
f0 = addfile("/mnt/d/shi/code1/data/1/ncepnov.t42l21.nc", "r")
uc300 = f0->u(0,{300},:,:)
f1 = addfile("/mnt/d/shi/code1/data/1/linear.t42l21.nc", "r")
z3001 = f1->z(nday1:nday2,{300},:,:)
u3001 = f1->u(nday1:nday2,{300},:,:)
;-------------------------------plot 1 经向风异常300百帕
v3001 = f1->v(nday1:nday2,{300},:,:)
;-------------------------------plot 2 500百帕hgt & uv
z5001 = f1->z(nday1:nday2,{500},:,:)
u5001 = f1->u(nday1:nday2,{500},:,:)
v5001 = f1->v(nday1:nday2,{500},:,:)
;-------------------------------plot 3 850百帕hgt & uv
z8501 = f1->z(nday1:nday2,{850},:,:)
u8501 = f1->u(nday1:nday2,{850},:,:)
v8501 = f1->v(nday1:nday2,{850},:,:)
;-------------------------------plot 4 SLP &uv
z10001 = f1->p(nday1:nday2,{1000},:,:);z(29,{1000},:,:)
u10001 = f1->u(nday1:nday2,{1000},:,:)
v10001 = f1->v(nday1:nday2,{1000},:,:)

;---------------------avg 
z300 = dim_avg_n_Wrap(z3001, 0)
u300 = dim_avg_n_Wrap(u3001, 0)
v300 = dim_avg_n_Wrap(v3001, 0)
z500 = dim_avg_n_Wrap(z5001, 0)
u500 = dim_avg_n_Wrap(u5001, 0)
v500 = dim_avg_n_Wrap(v5001, 0)
z850 = dim_avg_n_Wrap(z8501, 0)
u850 = dim_avg_n_Wrap(u8501, 0)
v850 = dim_avg_n_Wrap(v8501, 0)
z1000 = dim_avg_n_Wrap(z10001, 0)
u1000 = dim_avg_n_Wrap(u10001, 0)
v1000 = dim_avg_n_Wrap(v10001, 0)

f2 = addfile("/mnt/d/shi/code1/case2-lbm-data-v2-test1.nc","r")
Fx = f2->Fx1
Fy = f2->Fy1
f3 = addfile("/mnt/d/shi/code1/data/1/v1/frc.t42l21.nc","r")
tt1 = f3->t(0,:,:,:)
tt = tt1*86400;100000
copy_VarMeta(tt1, tt)

tp1 = tt({1000:200},{45:35},{0:30})
sum1 = new((/11/),"float")
num1 = new((/11/),"float")
t_profile = new((/11/),"float")
sum1 = 0
num1 = 0
do i = 0,10
    do j = 0,2
        do k = 0,10
            if(tp1(i,j,k).eq.0) then 
               sum1(i) = sum1(i) 
               num1(i) = num1(i) 
           else 
               sum1(i) = sum1(i) + tp1(i,j,k)
               num1(i) = num1(i) + 1
           end if 
       end do 
   end do 
end do 
do i = 0,10
   t_profile(i) = sum1(i)/28
end do
copy_VarMeta(tt(0:10,0,0), t_profile)


f4 = "/mnt/d/shi/code1/data/1/q1.txt"
y1 = asciiread(f4, (/11/), "float")
copy_VarMeta(tt(0:10,0,0), y1)

;==============================================ploting
wks = gsn_open_wks("x11","/mnt/d/shi/code1/2-lbm-frc-waf-test1-2-v2")

;-------------------------------plot1
nn = 0.01
nx = 2

zt = 0.01
bt = 2

resxy                   = True 
  resxy@gsnDraw           = False 
  resxy@gsnFrame          = False                    ; plot mods desired
  resxy@vpHeightF = 0.28 ;图的高度
  resxy@vpWidthF = 0.18
  resxy@vpXF     = 0.15
  resxy@vpYF     = 0.95
  resxy@xyDashPatterns = (/0,11/)
  resxy@xyLineThicknesses = (/2,2/)           
  resxy@xyLineColors      = (/"blue4","red3"/)          ; change line colors
  
  resxy@gsnXRefLine = 0
  resxy@gsnXRefLineColor = "gray25"
  resxy@gsnXRefLineDashPattern = "1"
  resxy@gsnXRefLineThicknesses = 1

  resxy@lbLabelFont   = "times-roman"
  resxy@tmXBLabelFont = "times-roman" 
  resxy@tmYLLabelFont = "times-roman"
  resxy@tiXAxisFont   = "times-roman"
  resxy@tiYAxisFont   = "times-roman"
  resxy@tiMainFont    = "times-roman"
  resxy@gsnStringFont = "times-roman"

  resxy@tmYROn = True
  resxy@tmYRLabelsOn = True
  resxy@tmYLOn = False
  resxy@tmXBMode = "Explicit"
  resxy@tmXBLabelFontHeightF = 0.01
  resxy@tmYRLabelFontHeightF = 0.01
  resxy@xyComputeYMin = True
  resxy@trYReverse    = True 
  ;resxy@xyYStyle   = "Log"
  resxy@trYMinF    = 200
  resxy@tmYRMode   = "Explicit"          
  resxy@tmYRValues = (/1000,950,900,850,700,600,500,400,300,250,200/)
  resxy@tmYRLabels = (/"1000","950","900","850","700","600","500","400","300","250","200"/)

  resxy@tmBorderThicknessF = 2
  resxy@tmXBMajorThicknessF = nx
  resxy@tmYRMajorThicknessF = nx
  resxy@tmXBMinorThicknessF = nx
  resxy@tmYRMinorThicknessF = nx
  resxy@tmXBMinorLengthF = 0.005
  resxy@tmXBMajorLengthF = 0.01
  resxy@tmXBMinorOutwardLengthF = 0.005
  resxy@tmXBMajorOutwardLengthF = 0.01
  resxy@tmYRMinorLengthF = 0.005
  resxy@tmYRMajorLengthF = 0.01
  resxy@tmYRMinorOutwardLengthF = 0.005
  resxy@tmYRMajorOutwardLengthF = 0.01

  resxy@pmLegendDisplayMode = "Always"
  resxy@pmLegendSide           = "Top"                ; Change location of 
  resxy@pmLegendParallelPosF   = 0.76               ; move units right
  resxy@pmLegendOrthogonalPosF = -1.07              ; move units down
  resxy@pmLegendWidthF         = 0.05                 ; Change width and
  resxy@pmLegendHeightF        = 0.031                  ; height of legend.
  resxy@lgLabelFontHeightF     = 0.01                   ; change font height
  resxy@lgPerimOn = True
  resxy@lgPerimThicknessF = 1

  name = (/" q1"," heating"/)
  resxy@xyExplicitLegendLabels = name
  resxy@lgLabelFontHeightF = 0.01
  resxy@lgLabelFont = "times-roman"

  
  resxy@gsnCenterString = "(a) K/Day"
  resxy@gsnCenterStringFontHeightF = nn
  resxy@gsnCenterStringOrthogonalPosF = 0.01

  resxy@tiYAxisOn   = True
  resxy@tiYAxisSide = "Right"
  resxy@tiYAxisString = "Pressure(hPa)"
  resxy@tiYAxisFontHeightF = nn
  resxy@tiYAxisOffsetXF = -0.005
  resxy@trXMinF = -0.3
  resxy@trXMaxF = 0.4
  y = new((/2,11/),"float")
  y(0,:) = y1
  y(1,:) = t_profile

plot1 = gsn_csm_xy(wks,y, y1&lev, resxy)

draw(plot1)

;----------------------------------plot2
;-------------- heating + 300waf + 500hgt
f5 = addfile("/mnt/d/dong/shi/code1/data/1/v1/frc.t42l21.nc","r")
tt2 = f5->t(0,{300},:,:)
ttt = tt2*86400;100000
copy_VarMeta(tt2, ttt)

  res                   = True   
  res@cnFillPalette     = "MPL_bwr"
  res@gsnAddCyclic      = True   
  res@gsnDraw           = False        
  res@gsnFrame          = False        
  res@gsnLeftString     = ""
  res@gsnRightString    = ""
  
  ; 过检区域变量=res
  resv = res
  sres = res
  vres = res
  vvres = res
  ; 设置经纬度范围                
  
  res@mpDataSetName = "Earth..4"
  res@mpGeophysicalLineColor = "gray25";"green4";"gray30"
  res@mpGeophysicalLineThicknessF = 2
  res@tmBorderThicknessF = bt
  res@tmXBMajorThicknessF = bt
  res@tmYLMajorThicknessF = bt
  res@tmXBMinorThicknessF = bt
  res@tmYLMinorThicknessF = bt

  lonR = 240
  lonL = 0
  latN = 80
  latS = 0

  res@mpMaxLatF = latN
  res@mpMinLatF = latS
  res@mpMaxLonF = lonR
  res@mpMinLonF = lonL

  res@tmXBMode = "Explicit"
  lon_value = (/0,30,60,90,120,150,180,210/)
  name1 = (/"","E","E","E","E","E","","W"/)
  lon_lable = new((/8/),"string")
  lon_lable(0:6) = abs(lon_value(0:6))+"~S~o~N~"+name1(0:6)
  lon_lable(7) = (lon_value(7)-60)+"~S~o~N~"+name1(7)

 
  lat_value = ispan(latS,latN,20)
  lat_lable = abs(lat_value)+"~S~o~N~N"
  res@tmYLValues = lat_value
  res@tmYLLabels = lat_lable
  res@tmXBLabelFontHeightF = zt
  res@tmYLMode = "Explicit"
  res@tmYLLabelFontHeightF = zt
  res@gsnLeftStringFontHeightF = zt
  res@tmXBValues = lon_value
  res@tmXBLabels = lon_lable

  res@tmXBMajorLengthF = 0.006
  res@tmXBMinorLengthF = 0.004
  res@tmXBMajorOutwardLengthF = 0.006
  res@tmXBMinorOutwardLengthF = 0.004
  res@tmYLMajorLengthF = 0.006
  res@tmYLMinorLengthF = 0.004
  res@tmYLMajorOutwardLengthF = 0.006
  res@tmYLMinorOutwardLengthF = 0.004
  res@tmXBLabelDeltaF = -0.8
  res@tmYLLabelDeltaF = -0.8
  
  res@lbLabelFont   = "times-roman"
  res@tmXBLabelFont = "times-roman" 
  res@tmYLLabelFont = "times-roman"
  res@tiXAxisFont   = "times-roman"
  res@tiYAxisFont   = "times-roman"
  res@tiMainFont    = "times-roman"
  res@gsnStringFont = "times-roman"
  res@cnInfoLabelOn = False      
  ;设置色标
  res@lbLabelBarOn     = False
  res@lbOrientation = "Horizontal"
  res@lbLabelFontHeightF = zt
  res@pmLabelBarWidthF = 0.8
  res@pmLabelBarHeightF = 0.08
  res@lbLabelStride = 1
  res@pmLabelBarOrthogonalPosF = 0.35
  res@lbFigureStringsFontHeightF = zt 
  res@lbBoxLinesOn = True
  res@lbBoxLineColor = "black"
  res@lbBoxLineThicknessF = bt

  ; 设置箭头
  refmag = 0.2 
  resv@vcGlyphStyle         = "Fillarrow"  
  resv@vcFillArrowEdgeColor = "red3";"black"
  resv@vcFillArrowEdgeThicknessF = 1
  resv@vcFillArrowFillColor = "red3"
  resv@vcFillArrowWidthF = 0.0005  
  resv@vcFillArrowHeadYF = 0.15   
  resv@vcFillArrowHeadXF = 0.3
  resv@vcFillArrowHeadInteriorXF = 0.015 
  resv@vcMinDistanceF  = 0.016;0.015    
  ; resv@vcMinMagnitudeF = 0.2;0.6  
  resv@vcMaxMagnitudeF = 1.2
  ;****设定参考箭头****
  resv@vcRefAnnoOn               = True  
  resv@vcRefMagnitudeF           = refmag  
  resv@vcRefLengthF              = 0.021 
  resv@vcRefAnnoBackgroundColor  = "white"      
  resv@vcRefAnnoPerimOn          = False       
  resv@vcRefAnnoFontHeightF      = 0.01        
  resv@vcRefAnnoString1On     = False           
  resv@vcRefAnnoString2On     = True    
  resv@vcRefAnnoString2       = refmag+"m/s"  
  resv@vcRefAnnoFont = "times-roman"
  resv@vcRefAnnoSide            = "Top" 
  resv@vcRefAnnoOrthogonalPosF  = -1.1 ;-0.435
  resv@vcRefAnnoParallelPosF    = 0.998  
  res@cnInfoLabelOn         = False      
  res@lbLabelBarOn          = True       

  resc = res
  resc@cnFillOn              = False      
  resc@cnLinesOn             = True
  resc@cnLineThicknessF      = 2        
  resc@cnLineColor = "green4";"gray25" 
  resc@gsnContourZeroLineThicknessF = 0. 
  resc@cnLineLabelsOn        = True   
  resc@cnLineDashPattern     = 2       
  resc@gsnContourNegLineDashPattern = 11;16
  resc@gsnContourPosLineDashPattern = 0
  resc@cnInfoLabelOn         = False 
; 画500高度场及过检区域
  cres = res
  cres@cnFillPalette = "WhiteYellowOrangeRed"
  cres@cnFillOn                 = True
  cres@cnLinesOn                = False
  cres@cnLevelSelectionMode     = "ExplicitLevels"
  ; cres@cnLevels                 = (/ispan(4, 40, 4)*0.1/)
;;---labelbar
  cres@lbLabelFontHeightF  = n
  cres@lbLabelBarOn     = False
  cres@lbOrientation = "vertically"
  cres@lbLabelFontHeightF = n
  cres@lbBoxLineThicknessF = b
  cres@pmLabelBarHeightF = 0.25
  cres@pmLabelBarWidthF = 0.05
  cres@pmLabelBarOrthogonalPosF = -1.01
  cres@pmLabelBarParallelPosF = 1.06
  cres@lbBoxSeparatorLinesOn    = True       
  cres@lbBoxLineThicknessF     = 2
  cres@lbBoxLinesOn             = True
  cres@lbLabelOffsetF = 0.2

  cc1 = cres 
  cc1@gsnLeftString = "(b) LBM"
  cc1@gsnLeftStringFontHeightF = zt
  cc1@gsnRightString = "500HGT & 300WAF"
  cc1@gsnRightStringFontHeightF = zt
  cc1@vpWidthF   = 0.43
  cc1@vpHeightF  = 0.2
  cc1@vpXF = 0.039 
  cc1@vpYF = 0.64
  plot2 = gsn_csm_contour_map(wks, ttt, cc1)

  cnn0 = resc
  plot3 = gsn_csm_contour(wks,z500,cnn0)
  overlay(plot2, plot3)

  ;---------------------------------------------------------------
vres@vcLineArrowThicknessF = 0.001
vres@vcFillArrowEdgeThicknessF = 1
vres@vcMinFracLengthF = 0.013
vres@vcMinDistanceF   = 0.018                  
vres@gsnLeftString=""
vres@gsnRightString=""
vres@vcGlyphStyle = "FillArrow"
vres@vcLineArrowColor     = "red3"
vres@vcFillArrowEdgeColor = "red3"
vres@vcFillArrowFillColor = "red3"
vres@vcFillArrowHeadXF = 0.4
vres@vcFillArrowHeadYF = 0.21
vres@vcFillArrowWidthF = 0.0001
vres@vcFillArrowHeadInteriorXF = 0.17;0.002
vres1 = vres
ref = 0.08
vres1@vcRefAnnoOn = True
vres1@vcRefAnnoSide = "Top"
vres1@vcRefAnnoFontHeightF = 0.01
vres1@vcRefAnnoOrthogonalPosF = -1.1
vres1@vcRefAnnoParallelPosF = 0.998
vres1@vcRefAnnoString1On = False
vres1@vcRefAnnoString2On = True
vres1@vcRefAnnoString2 = ref+"m~S~2~N~/"+"s~S~2~N"
vres1@vcRefAnnoPerimOn = False
vres1@vcRefMagnitudeF = ref 
vres1@vcRefLengthF = 0.035
vres1@vcMaxMagnitudeF = 0.15
; vres1@vcMinMagnitudeF = 0.015
vres1@vcRefAnnoFont = "times-roman"

x = where(Fx.ge.0, Fx, Fx@_FillValue)
copy_VarMeta(Fx, x)

b5 = Fy(:,{0:70}) 
b55 = where(b5.le.0,b5,b5@_FillValue)
copy_VarCoords(b5,b55)
Fy(:,{0:70}) = b55

plot4 = gsn_csm_vector(wks,x,Fy,vres1)
overlay(plot2, plot4)

  boxlat=(/50,50,40,40,50/)
  boxlon=(/200,225,225,200,200/)
  gonres =True
  gonres@gsLineColor      = "purple4"
  gonres@gsLineThicknessF = 5
  dum=gsn_add_polyline(wks, plot2,boxlon, boxlat, gonres)
draw(plot2)

;----------------------------------------------------plot3
  cc4 = cres 
  cc4@gsnLeftString = "(c) LBM"
  cc4@gsnLeftStringFontHeightF = zt
  cc4@gsnRightString = "SLP & UV1000"
  cc4@gsnRightStringFontHeightF = zt
  cc4@vpWidthF   = 0.43
  cc4@vpHeightF  = 0.2
  cc4@vpXF = 0.039  
  cc4@vpYF = 0.45
  cc4@lbLabelBarOn  = True
  cc4@lbOrientation = "Horizontal"
  cc4@lbLabelFontHeightF = zt
  cc4@pmLabelBarWidthF = 0.435
  cc4@pmLabelBarHeightF = 0.04
  cc4@pmLabelBarOrthogonalPosF = 0.13
  cc4@pmLabelBarParallelPosF = 0.5
  cc4@lbFigureStringsFontHeightF = zt 
  cc4@lbBoxLinesOn = True
  cc4@lbBoxLineColor = "black"
  cc4@lbBoxLineThicknessF = bt

  plot5 = gsn_csm_contour_map(wks, ttt, cc4)

  cnn2 = resc
  zz1000 = z1000*1
  copy_VarMeta(z1000, zz1000)
  zz1000@_FillVlue = -999

  plot6 = gsn_csm_contour(wks,zz1000,cnn2)
  plot7 = gsn_csm_vector(wks,u1000,v1000,resv)
  overlay(plot5, plot6)
  overlay(plot5, plot7)

  boxlat=(/50,50,40,40,50/)
  boxlon=(/200,225,225,200,200/)
  gonres =True
  gonres@gsLineColor      = "purple4"
  gonres@gsLineThicknessF = 5
  dum1=gsn_add_polyline(wks, plot5,boxlon, boxlat, gonres)
; print(v1000({0:20},{90}))
; exit
draw(plot5)


;=============================================================================
;----------------------- shiyan2
;=============================================================================
tx = 0.1
nday1 = 15
nday2 = 17
;-------------------------data 1 
f00 = addfile("/mnt/d/dong/shi/code1/data/1/ncepnov.t42l21.nc", "r")
ucx300 = f00->u(0,{300},:,:)
f11 = addfile("/mnt/d/dong/shi/code1/data/2/linear.t42l21.nc", "r")
zx3001 = f11->z(nday1:nday2,{300},:,:)
ux3001 = f11->u(nday1:nday2,{300},:,:)
;-------------------------------plot 1 经向风异常300百帕
vx3001 = f11->v(nday1:nday2,{300},:,:)
;-------------------------------plot 2 500百帕hgt & uv
zx5001 = f11->z(nday1:nday2,{500},:,:)
ux5001 = f11->u(nday1:nday2,{500},:,:)
vx5001 = f11->v(nday1:nday2,{500},:,:)
;-------------------------------plot 3 850百帕hgt & uv
zx8501 = f11->z(nday1:nday2,{850},:,:)
ux8501 = f11->u(nday1:nday2,{850},:,:)
vx8501 = f11->v(nday1:nday2,{850},:,:)
;-------------------------------plot 4 SLP &uv
zx10001 = f11->p(nday1:nday2,{1000},:,:);z(29,{1000},:,:)
ux10001 = f11->u(nday1:nday2,{1000},:,:)
vx10001 = f11->v(nday1:nday2,{1000},:,:)
; printVarSummary(z500)
; exit
;---------------------avg 
zx300 = dim_avg_n_Wrap(zx3001, 0)
ux300 = dim_avg_n_Wrap(ux3001, 0)
vx300 = dim_avg_n_Wrap(vx3001, 0)
zx500 = dim_avg_n_Wrap(zx5001, 0)
ux500 = dim_avg_n_Wrap(ux5001, 0)
vx500 = dim_avg_n_Wrap(vx5001, 0)
zx850 = dim_avg_n_Wrap(zx8501, 0)
ux850 = dim_avg_n_Wrap(ux8501, 0)
vx850 = dim_avg_n_Wrap(vx8501, 0)
zx1000 = dim_avg_n_Wrap(zx10001, 0)
ux1000 = dim_avg_n_Wrap(ux10001, 0)
vx1000 = dim_avg_n_Wrap(vx10001, 0)

f22 = addfile("/mnt/d/dong/shi/code1/case2-lbm-data-v2-test2.nc","r")
Fxx = f22->Fx1
Fyx = f22->Fy1
f3 = addfile("/mnt/d/dong/shi/code1/data/2/v1/frc.t42l21.nc","r")
ttx1 = f33->t(0,:,:,:)
ttx = tt1*86400;100000
copy_VarMeta(ttx1, ttx)

tpx1 = ttx({1000:200},{48:58},{330:355})

sumx1 = new((/11/),"float")
numx1 = new((/11/),"float")
t_profilex = new((/11/),"float")
sumx1 = 0
numx1 = 0
do i = 0,10
    do j = 0,3
        do k = 0,8
            if(tpx1(i,j,k).eq.0) then 
               sumx1(i) = sumx1(i) 
               numx1(i) = numx1(i) 
           else 
               sumx1(i) = sumx1(i) + tpx1(i,j,k)
               numx1(i) = numx1(i) + 1
           end if 
       end do 
   end do 
end do 
do i = 0,10
   t_profilex(i) = sumx1(i)/28.
end do
copy_VarMeta(ttx(0:10,0,0), t_profilex)


f44 = "/mnt/d/dong/shi/code1/data/2/q1.txt"
yx1 = asciiread(f44, (/11/), "float")
copy_VarMeta(ttx(0:10,0,0), yx1)


;-------------------------data 2
f00 = addfile("/mnt/d/dong/shi/code1/data/1/ncepnov.t42l21.nc", "r")
ucx300 = f00->u(0,{300},:,:)
f11 = addfile("/mnt/d/dong/shi/code1/data/2/linear.t42l21.nc", "r")
zx3001 = f11->z(nday1:nday2,{300},:,:)
ux3001 = f11->u(nday1:nday2,{300},:,:)
;-------------------------------plot 1 经向风异常300百帕
vx3001 = f11->v(nday1:nday2,{300},:,:)
;-------------------------------plot 2 500百帕hgt & uv
zx5001 = f11->z(nday1:nday2,{500},:,:)
ux5001 = f11->u(nday1:nday2,{500},:,:)
vx5001 = f11->v(nday1:nday2,{500},:,:)
;-------------------------------plot 3 850百帕hgt & uv
zx8501 = f11->z(nday1:nday2,{850},:,:)
ux8501 = f11->u(nday1:nday2,{850},:,:)
vx8501 = f11->v(nday1:nday2,{850},:,:)
;-------------------------------plot 4 SLP &uv
zx10001 = f11->p(nday1:nday2,{1000},:,:);z(29,{1000},:,:)
ux10001 = f11->u(nday1:nday2,{1000},:,:)
vx10001 = f11->v(nday1:nday2,{1000},:,:)
; printVarSummary(z500)
; exit
;---------------------avg 
zx300 = dim_avg_n_Wrap(zx3001, 0)
ux300 = dim_avg_n_Wrap(ux3001, 0)
vx300 = dim_avg_n_Wrap(vx3001, 0)
zx500 = dim_avg_n_Wrap(zx5001, 0)
ux500 = dim_avg_n_Wrap(ux5001, 0)
vx500 = dim_avg_n_Wrap(vx5001, 0)
zx850 = dim_avg_n_Wrap(zx8501, 0)
ux850 = dim_avg_n_Wrap(ux8501, 0)
vx850 = dim_avg_n_Wrap(vx8501, 0)
zx1000 = dim_avg_n_Wrap(zx10001, 0)
ux1000 = dim_avg_n_Wrap(ux10001, 0)
vx1000 = dim_avg_n_Wrap(vx10001, 0)
; printVarSummary(z500)
; exit
f22 = addfile("/mnt/d/dong/shi/code1/case2-lbm-data-v2-test2.nc","r")
Fxx = f22->Fx1
Fyx = f22->Fy1
f33 = addfile("/mnt/d/dong/shi/code1/data/2/v1/frc.t42l21.nc","r")
ttx1 = f33->t(0,:,:,:)
ttx = ttx1*86400;100000
copy_VarMeta(ttx1, ttx)

tpx1 = ttx({1000:200},{48:58},{330:355})

sumx1 = new((/11/),"float")
numx1 = new((/11/),"float")
t_profilex = new((/11/),"float")
sumx1 = 0
numx1 = 0
do i = 0,10
    do j = 0,3
        do k = 0,8
            if(tpx1(i,j,k).eq.0) then 
               sumx1(i) = sumx1(i) 
               numx1(i) = numx1(i) 
           else 
               sumx1(i) = sumx1(i) + tpx1(i,j,k)
               numx1(i) = numx1(i) + 1
           end if 
       end do 
   end do 
end do 
do i = 0,10
   t_profilex(i) = sumx1(i)/28.
end do
copy_VarMeta(ttx(0:10,0,0), t_profilex)


f44 = "/mnt/d/dong/shi/code1/data/2/q1.txt"
yx1 = asciiread(f44, (/11/), "float")
copy_VarMeta(ttx(0:10,0,0), yx1)

;==============================================ploting
;-------------------------------plot1
nn = 0.01
nx = 2

zt = 0.01
bt = 2

  resx                   = True 
  resx@gsnDraw           = False 
  resx@gsnFrame          = False  
  ; resxy@vpHeightF = 0.28 ;图的高度
  ; resxy@vpWidthF = 0.18
  ; resxy@vpXF     = 0.15
  ; resxy@vpYF     = 0.95                  ; plot mods desired
  resx@vpHeightF = 0.28 ;图的高度
  resx@vpWidthF = 0.21
  resx@vpXF     = 0.62
  resx@vpYF     = 0.95
  resx@xyDashPatterns = (/0,11/)
  resx@xyLineThicknesses = (/2,2/)           
  resx@xyLineColors      = (/"blue4","red3"/)          ; change line colors
  
  resx@gsnXRefLine = 0
  resx@gsnXRefLineColor = "gray25"
  resx@gsnXRefLineDashPattern = "1"
  resx@gsnXRefLineThicknesses = 1

  resx@lbLabelFont   = "times-roman"
  resx@tmXBLabelFont = "times-roman" 
  resx@tmYLLabelFont = "times-roman"
  resx@tiXAxisFont   = "times-roman"
  resx@tiYAxisFont   = "times-roman"
  resx@tiMainFont    = "times-roman"
  resx@gsnStringFont = "times-roman"

  resx@tmYROn = True
  resx@tmYRLabelsOn = True
  resx@tmYLOn = False
  resx@tmXBMode = "Explicit"
  resx@tmXBLabelFontHeightF = 0.01
  resx@tmYRLabelFontHeightF = 0.01
  resx@xyComputeYMin = True
  resx@trYReverse    = True 
  ;resx@xyYStyle   = "Log"
  resx@trYMinF    = 200
  resx@tmYRMode   = "Explicit"          
  resx@tmYRValues = (/1000,950,900,850,700,600,500,400,300,250,200/)
  resx@tmYRLabels = (/"1000","950","900","850","700","600","500","400","300","250","200"/)

  resx@tmBorderThicknessF = 2
  resx@tmXBMajorThicknessF = nx
  resx@tmYRMajorThicknessF = nx
  resx@tmXBMinorThicknessF = nx
  resx@tmYRMinorThicknessF = nx
  resx@tmXBMinorLengthF = 0.005
  resx@tmXBMajorLengthF = 0.01
  resx@tmXBMinorOutwardLengthF = 0.005
  resx@tmXBMajorOutwardLengthF = 0.01
  resx@tmYRMinorLengthF = 0.005
  resx@tmYRMajorLengthF = 0.01
  resx@tmYRMinorOutwardLengthF = 0.005
  resx@tmYRMajorOutwardLengthF = 0.01

  resx@pmLegendDisplayMode = "Always"
  resx@pmLegendSide           = "Top"                ; Change location of 
  resx@pmLegendParallelPosF   = 0.795              ; move units right
  resx@pmLegendOrthogonalPosF = -1.07              ; move units down
  resx@pmLegendWidthF         = 0.05                 ; Change width and
  resx@pmLegendHeightF        = 0.031                  ; height of legend.
  resx@lgLabelFontHeightF     = 0.01                   ; change font height
  resx@lgPerimOn = True
  resx@lgPerimThicknessF = 1

  namex = (/" q1"," heating"/)
  resx@xyExplicitLegendLabels = namex
  resx@lgLabelFontHeightF = 0.01
  resx@lgLabelFont = "times-roman"

  
  resx@gsnCenterString = "(d) K/Day"
  resx@gsnCenterStringFontHeightF = nn
  resx@gsnCenterStringOrthogonalPosF = 0.01

  resx@tiYAxisOn   = True
  resx@tiYAxisSide = "Right"
  resx@tiYAxisString = "Pressure(hPa)"
  resx@tiYAxisFontHeightF = nn
  resx@tiYAxisOffsetXF = -0.005
  resx@trXMinF = -0.8
  resx@trXMaxF = 0.2

  yx = new((/2,11/),"float")
  yx(0,:) = yx1
  yx(1,:) = t_profilex

plot11 = gsn_csm_xy(wks,yx, yx1&lev, resx)

draw(plot11)

;----------------------------------plot2
;-------------- heating + 300waf + 500hgt
f55 = addfile("/mnt/d/dong/shi/code1/data/2/v1/frc.t42l21.nc","r")
ttx2 = f55->t(0,{300},:,:)
tttx = ttx2*86400;100000
copy_VarMeta(ttx2, tttx)

  re                   = True   
  re@cnFillPalette     = "MPL_bwr"
  re@gsnAddCyclic      = True   
  re@gsnDraw           = False        
  re@gsnFrame          = False        
  re@gsnLeftString     = ""
  re@gsnRightString    = ""
  re@mpLandFillColor        = 0
  ; 过检区域变量=res
  rev = re
  sre = re
  vre = re
  vvre = re
  ; 设置经纬度范围                
  
  re@mpDataSetName = "Earth..4"
  re@mpGeophysicalLineColor = "gray25";"green4";"gray30"
  re@mpGeophysicalLineThicknessF = 2
  re@tmBorderThicknessF = bt
  re@tmXBMajorThicknessF = bt
  re@tmYLMajorThicknessF = bt
  re@tmXBMinorThicknessF = bt
  re@tmYLMinorThicknessF = bt

  lonR1 = 240
  lonL1 = -30
  latN1 = 80
  latS1 = 0

  re@mpMaxLatF = latN1
  re@mpMinLatF = latS1
  re@mpMaxLonF = lonR1
  re@mpMinLonF = lonL1

  re@tmXBMode = "Explicit"
  lon_value1 = (/-30,0,30,60,90,120,150,180,210/)
  namex1 = (/"W","","E","E","E","E","E","","W"/)
  lon_lable1 = new((/9/),"string")
  lon_lable1(0:7) = abs(lon_value1(0:7))+"~S~o~N~"+namex1(0:7)
  lon_lable1(8) = (lon_value1(8)-60)+"~S~o~N~"+namex1(8)

  lat_value1 = ispan(latS1,latN1,20)
  lat_lable1 = abs(lat_value1)+"~S~o~N~N"

  re@tmYLValues = lat_value1
  re@tmYLLabels = lat_lable1

  re@tmXBLabelFontHeightF = zt
  re@tmYLMode = "Explicit"
  re@tmYLLabelFontHeightF = zt
  re@gsnLeftStringFontHeightF = zt
  re@tmXBValues = lon_value
  re@tmXBLabels = lon_lable

  re@tmXBMajorLengthF = 0.006
  re@tmXBMinorLengthF = 0.004
  re@tmXBMajorOutwardLengthF = 0.006
  re@tmXBMinorOutwardLengthF = 0.004
  re@tmYLMajorLengthF = 0.006
  re@tmYLMinorLengthF = 0.004
  re@tmYLMajorOutwardLengthF = 0.006
  re@tmYLMinorOutwardLengthF = 0.004
  re@tmXBLabelDeltaF = -0.8
  re@tmYLLabelDeltaF = -0.8
  
  re@lbLabelFont   = "times-roman"
  re@tmXBLabelFont = "times-roman" 
  re@tmYLLabelFont = "times-roman"
  re@tiXAxisFont   = "times-roman"
  re@tiYAxisFont   = "times-roman"
  re@tiMainFont    = "times-roman"
  re@gsnStringFont = "times-roman"
  re@cnInfoLabelOn = False       
  ;设置色标
  re@lbLabelBarOn     = False
  re@lbOrientation = "Horizontal"
  re@lbLabelFontHeightF = zt
  re@pmLabelBarWidthF = 0.8
  re@pmLabelBarHeightF = 0.08
  re@lbLabelStride = 1
  re@pmLabelBarOrthogonalPosF = 0.35
  re@lbFigureStringsFontHeightF = zt 
  re@lbBoxLinesOn = True
  re@lbBoxLineColor = "black"
  re@lbBoxLineThicknessF = bt

  ; 设置箭头
  refmag1 = 0.2 
  rev@vcGlyphStyle         = "Fillarrow"  
  rev@vcFillArrowEdgeColor = "red3";"black"
  rev@vcFillArrowEdgeThicknessF = 1
  rev@vcFillArrowFillColor = "red3"
  rev@vcFillArrowWidthF = 0.0005   
  rev@vcFillArrowHeadYF = 0.15   
  rev@vcFillArrowHeadXF = 0.3
  rev@vcFillArrowHeadInteriorXF = 0.015 
  rev@vcMinDistanceF  = 0.012;0.018    
  rev@vcMaxMagnitudeF = 0.9;0.6  
  ;****设定参考箭头****
  rev@vcRefAnnoOn               = True  
  rev@vcRefMagnitudeF           = refmag1  
  rev@vcRefLengthF              = 0.02 
  rev@vcRefAnnoBackgroundColor  = "white"     
  rev@vcRefAnnoPerimOn          = False    
  rev@vcRefAnnoFontHeightF      = 0.01        
  rev@vcRefAnnoString1On     = False           
  rev@vcRefAnnoString2On     = True    
  rev@vcRefAnnoString2       = refmag+"m/s"  
  rev@vcRefAnnoFont = "times-roman"
  rev@vcRefAnnoSide            = "Top" 
  rev@vcRefAnnoOrthogonalPosF  = -1.1 ;-0.435
  rev@vcRefAnnoParallelPosF    = 0.998  
  re@cnInfoLabelOn         = False       
  re@lbLabelBarOn          = True       
  
  
  rec = re
  rec@cnFillOn              = False     
  rec@cnLinesOn             = True
  rec@cnLineThicknessF      = 2       
  rec@cnLineColor = "green4";"gray25" 
  rec@gsnContourZeroLineThicknessF = 0. 
  rec@cnLineLabelsOn        = True   
  rec@cnLineDashPattern     = 2      
  rec@gsnContourNegLineDashPattern = 11;16
  rec@gsnContourPosLineDashPattern = 0
  rec@cnInfoLabelOn         = False
 

; 画500高度场及过检区域
  cre = re
  cre@cnFillPalette = "GMT_ocean";"topo_15lev";"WhiteYellowOrangeRed"
  cre@cnFillOn                 = True
  cre@cnLinesOn                = False
  cre@cnLevelSelectionMode     = "ExplicitLevels"
  cre@cnLevels  = (/ispan(-70, -15, 5)*0.01/)
;;---labelbar
  cre@lbLabelFontHeightF       = n
  cre@lbLabelBarOn     = False
  cre@lbOrientation = "vertically"
  cre@lbLabelFontHeightF = n
  cre@lbBoxLineThicknessF = b
  cre@pmLabelBarHeightF = 0.25
  cre@pmLabelBarWidthF = 0.05
  cre@pmLabelBarOrthogonalPosF = -1.01
  cre@pmLabelBarParallelPosF = 1.06
  cre@lbBoxSeparatorLinesOn    = True       
  cre@lbBoxLineThicknessF     = 2
  cre@lbBoxLinesOn             = True
  cre@lbLabelOffsetF = 0.2

  ccx1 = cre 
  ccx1@gsnLeftString = "(e) LBM"
  ccx1@gsnLeftStringFontHeightF = zt
  ccx1@gsnRightString = "500HGT & 300WAF"
  ccx1@gsnRightStringFontHeightF = zt
  ccx1@vpWidthF   = 0.47
  ; ccx1@vpHeightF  = 0.3
  ccx1@vpXF = 0.52 
  ccx1@vpYF = 0.66

  tttx({0:40},{-30:180}) = tttx@_FillValue
  tttx({0:90},{0:180}) = tttx@_FillValue
  tttx({60:80},{-30:0}) = tttx@_FillValue
  frc1 = tttx
  copy_VarMeta(tttx, frc1)

  plot22 = gsn_csm_contour_map(wks, frc1, ccx1)

  cnnx0 = rec
  cnnx0@cnLevelSelectionMode  = "ExplicitLevels" 
  cnnx0@cnLevels  = (/-14,-12,-10,-8,-6,-4,-2,0,1,2,3,4,5,6,7/)
  ; cnnx0@cnLevels  = (/ispan(-5,30,1)*1/)
  plot33 = gsn_csm_contour(wks,zx500,cnnx0)
  overlay(plot22, plot33)

  ;---------------------------------------------------------------
vre@vcLineArrowThicknessF = 0.001
vre@vcFillArrowEdgeThicknessF = 1
vre@vcMinFracLengthF = 0.018
vre@vcMinDistanceF   = 0.018;0.018                  
vre@gsnLeftString=""
vre@gsnRightString=""
vre@vcGlyphStyle = "FillArrow"
vre@vcLineArrowColor     = "red3"
vre@vcFillArrowEdgeColor = "red3"
vre@vcFillArrowFillColor = "red3"
vre@vcFillArrowHeadXF = 0.4
vre@vcFillArrowHeadYF = 0.21
vre@vcFillArrowWidthF = 0.0001
vre@vcFillArrowHeadInteriorXF = 0.17;0.002
vre1 = vre
ref1 = 0.05
vre1@vcRefAnnoOn = True
vre1@vcRefAnnoSide = "Top"
vre1@vcRefAnnoFontHeightF = 0.01
vre1@vcRefAnnoOrthogonalPosF = -1.1
vre1@vcRefAnnoParallelPosF = 0.998
vre1@vcRefAnnoString1On = False
vre1@vcRefAnnoString2On = True
vre1@vcRefAnnoString2 = ref1+"m~S~2~N~/"+"s~S~2~N"
vre1@vcRefAnnoPerimOn = False
vre1@vcRefMagnitudeF = ref1 
vre1@vcRefLengthF = 0.04
vre1@vcMaxMagnitudeF = 0.12
; vre1@vcMinMagnitudeF = 0.015
vre1@vcRefAnnoFont = "times-roman"

xx = where(Fxx.ge.0, Fxx, Fxx@_FillValue)
copy_VarMeta(Fxx, xx)

a5 = Fyx(:,{-30:50}) 
a55 = where(a5.ge.0,a5,a5@_FillValue)
copy_VarCoords(a5,a55)
Fyx(:,{-30:50}) = a55


plot44 = gsn_csm_vector(wks,xx,Fyx,vre1)
overlay(plot22, plot44)

  boxlat=(/50,50,40,40,50/)
  boxlon=(/200,225,225,200,200/)
  gonres1 =True
  gonres1@gsLineColor      = "purple4"
  gonres1@gsLineThicknessF = 5
  dumx = gsn_add_polyline(wks, plot22,boxlon, boxlat, gonres1)
draw(plot22)

;----------------------------------------------------plot3
  ccx4 = cre 
  ccx4@gsnLeftString = "(f) LBM"
  ccx4@gsnLeftStringFontHeightF = zt
  ccx4@gsnRightString = "SLP & UV1000"
  ccx4@gsnRightStringFontHeightF = zt
  ccx4@vpWidthF   = 0.47
  ccx4@vpXF = 0.52 
  ccx4@vpYF = 0.47
  ccx4@lbLabelBarOn  = True
  ccx4@lbOrientation = "Horizontal"
  ccx4@lbLabelFontHeightF = zt
  ccx4@pmLabelBarWidthF = 0.475
  ccx4@pmLabelBarHeightF = 0.04
  ccx4@pmLabelBarOrthogonalPosF = 0.13
  ccx4@pmLabelBarParallelPosF = 0.5
  ccx4@lbFigureStringsFontHeightF = zt 
  ccx4@lbBoxLinesOn = True
  ccx4@lbBoxLineColor = "black"
  ccx4@lbBoxLineThicknessF = bt

  plot55 = gsn_csm_contour_map(wks, tttx, ccx4)

  cnnx2 = rec
  cnnx2@cnLevelSelectionMode  = "ExplicitLevels" 
  cnnx2@cnLevels  = (/-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.5,1,1.5,2,2.5,3,3.5/)
  zzx1000 = zx1000*1
  copy_VarMeta(zx1000, zzx1000)
  zzx1000@_FillVlue = -999

  plot66 = gsn_csm_contour(wks,zzx1000,cnnx2)
  plot77 = gsn_csm_vector(wks,ux1000,vx1000,rev)
  overlay(plot55, plot66)
  overlay(plot55, plot77)

  boxlat=(/50,50,40,40,50/)
  boxlon=(/200,225,225,200,200/)
  gonres1 =True
  gonres1@gsLineColor      = "purple4"
  gonres1@gsLineThicknessF = 5
  dumx1=gsn_add_polyline(wks, plot55,boxlon, boxlat, gonres1)
; print(v1000({0:20},{90}))
; exit
draw(plot55)
frame(wks)
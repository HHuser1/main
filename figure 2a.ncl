
;===========================================================
;-------------- WAF & vwnd & uclim
;===========================================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"        ; functions required to
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"             ; plot.  include before
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"  


;----- data
; f1 = addfile ("/mnt/d/shi/code1/Nov_13winterblob_data.nc", "r")
f1 = addfile ("/mnt/d/shi/code1/Oct_13winterblob_data.nc", "r")
Fx1 = f1->waf_x(::-1,:)
Fy1 = f1->waf_y(::-1,:)
u1 = f1->long_term_mean_uwnd(::-1,:)
v1 = f1->vwnd_anomaly_13WinterBlob(::-1,:)
vtest1 = f1->vwnd_ttest_pvalue(::-1,:)
Fx = dble2flt(Fx1)
Fy = dble2flt(Fy1)
u = dble2flt(u1)
v = dble2flt(v1)
vtest = dble2flt(vtest1)
; print(vtest)
; exit
f4 = addfile ("/mnt/f/hgt.2021.nc", "r")                      
hc = f4->hgt(4,{300},{87.5:0},{0:270})
hhc = hc(::-1,:)
copy_VarMeta(hhc, Fx(:,{0:270}))
copy_VarMeta(hhc, Fy(:,{0:270}))
copy_VarMeta(hhc, u(:,{0:270}))
copy_VarMeta(hhc, v(:,{0:270}))
copy_VarMeta(hhc, vtest(:,{0:270}))

hc1 = f4->hgt(4,{300},{87.5:0},{330:360})
hhc1 = hc1(::-1,:)
copy_VarMeta(hhc1, Fx(:,12:24))
copy_VarMeta(hhc1, Fy(:,12:24))
copy_VarMeta(hhc1, u(:,12:24))
copy_VarMeta(hhc1, v(:,12:24))
copy_VarMeta(hhc1, vtest(:,12:24))
; printVarSummary(hhc1)
; printVarSummary(Fx(:,12:24))
; ; print(vtest)
; exit
;==============================================ploting
zt = 0.012
bt = 1
tx = 0.1
wks_type = "x11" 
wks = gsn_open_wks(wks_type,"/mnt/d/shi/code1/4-waf-v-u-oct")

res                   = True   
res@cnFillPalette     = "NCV_blu_red";"MPL_bwr"
res@gsnAddCyclic      = False   
res@gsnDraw           = False        
res@gsnFrame          = False        
res@gsnLeftString     = ""
res@gsnRightString    = ""
  
  ; 过检区域变量=res
  sres = res
  vres = res
  cres = res
  ; 设置经纬度范围                
  
  res@mpDataSetName = "Earth..4"
  res@mpGeophysicalLineColor = "gray25";"green4";"gray30"
  res@mpGeophysicalLineThicknessF = 2
  res@tmBorderThicknessF = bt
  res@tmXBMajorThicknessF = bt
  res@tmYLMajorThicknessF = bt
  res@tmXBMinorThicknessF = bt
  res@tmYLMinorThicknessF = bt


  lonR = 270
  lonL = -30
  latN = 87.5
  latS = 0

  res@mpMaxLatF = latN
  res@mpMinLatF = latS
  res@mpMaxLonF = lonR
  res@mpMinLonF = lonL

  res@tmXBMode = "Explicit"
  lon_value = (/-30,30,90,150,210,270/)
  name1 = (/"W","E","E","E","W","W"/)
  lon_lable = new((/6/),"string")
  lon_lable(0:3) = abs(lon_value(0:3))+"~S~o~N~"+name1(0:3)
  lon_lable(4) = abs(lon_value(4)-60)+"~S~o~N~"+name1(4)
  lon_lable(5) = abs(lon_value(5)-180)+"~S~o~N~"+name1(5)
 
  ; lat_value = ispan(latS,latN,10)
  ; lat_lable = abs(lat_value)+"~S~o~N~N"

  lat_value = (/20,40,60,80/)
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
  
  ; res@lbLabelFont   = "times-roman"
  ; res@tmXBLabelFont = "times-roman" 
  ; res@tmYLLabelFont = "times-roman"
  ; res@tiXAxisFont   = "times-roman"
  ; res@tiYAxisFont   = "times-roman"
  ; res@tiMainFont    = "times-roman"
  ; res@gsnStringFont = "times-roman"
  res@cnInfoLabelOn = False       
  ;设置色标
  res@lbLabelBarOn     = True
  res@lbOrientation = "Vertical"
  res@lbLabelFontHeightF = zt
  res@pmLabelBarWidthF = 0.06
  res@pmLabelBarHeightF = 0.225
  res@lbLabelStride = 1
  res@pmLabelBarOrthogonalPosF = 0.0001
  res@lbFigureStringsFontHeightF = zt 
  res@lbBoxLinesOn = True
  res@lbBoxLineColor = "black"
  res@lbBoxLineThicknessF = bt

;----------------------------- v300 
  resc = res
  sres@cnFillOn             = True         
  sres@cnLinesOn            = False         
  sres@cnLineColor          = "gray"      
  sres@cnLineThicknessF     = 0.3         
  sres@cnLineLabelsOn       = False       
  sres@lbLabelBarOn = False
  sres@cnInfoLabelOn        = False
  sres@cnLevelSelectionMode = "ExplicitLevels" 
  sres@cnLevels = (/0.1/)                                                                                                                                                                                             
  sres@cnMonoFillPattern    = False                                                                          
  sres@cnFillPatterns       = (/17,-1/)                                                                             
  sres@cnMonoFillScale      = False                                                                         
  sres@cnFillScales         = (/1/)    ; change densities                                                                          
  sres@cnMonoFillColor      =True 
  sres@cnFillColor = "white" ;  "red";                                                                                             
  sres@cnFillDotSizeF       = 0.0035 


  resc@cnLevelSelectionMode  = "ExplicitLevels"                
  ; resc@cnLevels  = (/ispan(-60,60,5)*1/)  ;NCEP  
  resc@cnLevels  = (/ispan(-50,50,5)*0.1/)  ;ERA5 
  resc@cnFillOn              = True     
  resc@cnLinesOn             = False
  resc@cnLineThicknessF      = 1.5         
  resc@gsnContourZeroLineThicknessF = 0. 
  resc@cnLineLabelsOn        = False    
  resc@cnLineDashPattern     = 2       
  resc@cnInfoLabelOn         = False 

  cnn = resc
  cnnn = resc
  ;------uclim
  cres@cnLevelSelectionMode  = "ExplicitLevels" 
  cres@cnLevels  = (/20,30,40,50,60,70/)
  cres@cnLineThicknessF      = 4   
  cres@cnLineColor = "green2";gray29" 
  cres@gsnContourZeroLineThicknessF = 0. 
  cres@cnLineLabelsOn        = True  
  cres@cnLineLabelFontHeightF = 0.006
  cres@cnLineDashPattern     = 0       
  cres@cnInfoLabelOn         = False 
  cres@cnFillOn              = False      
  cres@cnLinesOn             = True
  cres@gsnContourPosLineDashPattern = 0
  cres@gsnContourNegLineDashPattern = 1
; 画500高度场及过检区域
; resc@gsnLeftString = "(a) Nov. WAF&vwnd&u_clim at 300hPa"
  resc@gsnLeftString = "(a) Oct. WAF&vwnd&u_clim at 300hPa"
  resc@gsnLeftStringFontHeightF    = zt
  resc@gsnLeftStringOrthogonalPosF = 0.035

  base = gsn_csm_contour_map(wks,v,resc)
  plot1 = gsn_csm_contour(wks,vtest,sres)
  overlay(base, plot1)


  base1 = gsn_csm_contour(wks,u,cres)
  overlay(base, base1)

;---------------------------------------------------------------
vres@vcLineArrowThicknessF = 3;0.12
vres@vcMinFracLengthF = 0.005
vres@vcMinDistanceF   = 0.021 ;0.02                 
vres@gsnLeftString=""
vres@gsnRightString=""

vres@vcGlyphStyle = "FillArrow"
vres@vcLineArrowColor     = "gray10"
vres@vcFillArrowEdgeColor = "gray10"
vres@vcFillArrowFillColor = "gray10"
vres@vcFillArrowHeadXF = 0.3
vres@vcFillArrowHeadYF = 0.15
vres@vcFillArrowWidthF = 0.005
vres@vcFillArrowHeadInteriorXF = 0.13;0.01

vres1 = vres
vec = 3
vres1@vcRefAnnoOn = True
vres1@vcRefAnnoSide = "Top"
vres1@vcRefAnnoFontHeightF = 0.0075
vres1@vcRefAnnoOrthogonalPosF = -1.11;-0.315
vres1@vcRefAnnoParallelPosF = 0.9982
vres1@vcRefAnnoString1On = True
vres1@vcRefAnnoString2On = False
vres1@vcRefAnnoString1 = vec+"m~S~2~N~/"+"s~S~2~N"
vres1@vcRefAnnoPerimOn = False
vres1@vcRefMagnitudeF = vec 
vres1@vcRefLengthF = 0.02
; vres1@vcRefAnnoFont = "times-roman"
vres1@vcMaxMagnitudeF = 98
vres1@vcMinMagnitudeF = 0.8

; vv4 = where(abs(Fy).le.160, Fy, Fy@_FillValue)
; uu4 = where(Fx.ge.0.and.Fx.le.180, Fx, Fx@_FillValue)

zt = 0.035
vres11 = vres1
; vres11@gsnLeftString = "(a) Oct."
vres11@gsnLeftStringFontHeightF = zt
plot = gsn_csm_vector(wks,Fx,Fy,vres11)
overlay(base, plot)

draw(base)
frame(wks)

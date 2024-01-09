
begin
;----- Select level for Calculate
diri = "/mnt/d/shi/code1/data/"
;---- monthly mean data
filename = systemfunc("ls "+diri+"hgt.mon.mean.nc")
zfile = addfile(filename,"r")
filename = systemfunc("ls "+diri+"uwnd.mon.mean.nc")
ufile = addfile(filename,"r")
filename = systemfunc("ls "+diri+"vwnd.mon.mean.nc")
vfile = addfile(filename,"r")
filename = systemfunc("ls "+diri+"slp.mon.mean.nc")
slpfile = addfile(filename,"r")
;---------------------------------- 全时段
time = zfile->time
YYYYMM = cd_calendar(time,-1)
it_s = 195801
it_e = 199712
rec_s = ind(it_s .eq. YYYYMM)
rec_e = ind(it_e .eq. YYYYMM)
delete(YYYYMM)
;-------------------------------------------------------------
;------------- 300hgt
; monthly mean data
if (getfilevartypes(zfile,"hgt").eq."short")then
z1= short2flt(zfile->hgt(rec_s:rec_e,{300},:,:))
else
z1 = zfile->hgt(rec_s:rec_e,{300},:,:)
end if
;-------------- climate 
cz1 = clmMonTLL(z1)  ;12*73*144

;-------------- 300u
if (getfilevartypes(ufile,"uwnd").eq."short")then
u1 = short2flt(ufile->uwnd(rec_s:rec_e,{300},:,:))
else
u1 = ufile->uwnd(rec_s:rec_e,{300},:,:)
end if
;-------------- climate 
cu1 = clmMonTLL(u1)  ;12*73*144
;-----------------------------------------------------------
;------------- 500hgt
if (getfilevartypes(zfile,"hgt").eq."short")then
z2 = short2flt(zfile->hgt(rec_s:rec_e,{500},:,:))
else
z2 = zfile->hgt(rec_s:rec_e,{500},:,:)
end if

;-------------- climate 
cz2 = clmMonTLL(z2)  ;12*73*144
;------------------------------------------------------------
;-------------- slp
if (getfilevartypes(slpfile,"slp").eq."short")then
slp1 = short2flt(slpfile->slp(rec_s:rec_e,:,:))
else
slp1 = slpfile->slp(rec_s:rec_e,:,:)
end if

;-------------- climate 
cslp = clmMonTLL(slp1)  ;12*73*144

;-------------- 1000uv
if (getfilevartypes(ufile,"uwnd").eq."short")then
u2 = short2flt(ufile->uwnd(rec_s:rec_e,{1000},:,:))
else
u2 = ufile->uwnd(rec_s:rec_e,{1000},:,:)
end if
;-------------- climate 
cu2 = clmMonTLL(u2)

if (getfilevartypes(vfile,"vwnd").eq."short")then
v2 = short2flt(vfile->vwnd(rec_s:rec_e,{1000},:,:))
else
v2 = vfile->vwnd(rec_s:rec_e,{1000},:,:)
end if
;-------------- climate 
cv2 = clmMonTLL(v2)

;===================================================plot 1 
 wks = gsn_open_wks("x11","/mnt/d/shi/code1/3-clim")
  zt = 0.013
  bt = 3
  res                   = True   
  res@gsnAddCyclic      = True   
  res@gsnDraw           = False        
  res@gsnFrame          = False        
  res@gsnLeftString     = ""
  res@gsnRightString    = ""
  res@cnFillOn              = True      
  res@cnLinesOn             = False
  res@cnFillPalette     = "MPL_bwr"
  ; 过检区域变量=res
  resv = res
  sres = res
  cres1 = res
  ; 设置经纬度范围                 

  res@vpXF             = 0.11                 
  res@vpYF             = 0.98                  
  res@vpWidthF         = 0.85
  res@vpHeightF        = 0.35 

  res@mpDataSetName = "Earth..4"
  res@mpGeophysicalLineColor = "gray60";"gray30"
  res@mpGeophysicalLineThicknessF = 2
  res@tmBorderThicknessF = bt
  res@tmXBMajorThicknessF = bt
  res@tmYLMajorThicknessF = bt
  res@tmXBMinorThicknessF = bt
  res@tmYLMinorThicknessF = bt

  lonR = 180
  lonL = -180
  latN = 80
  latS = 0

  res@mpMaxLatF = latN
  res@mpMinLatF = latS
  ; res@mpMaxLonF = lonR
  ; res@mpMinLonF = lonL
  res@mpCenterLonF = 120
  
  res@tmXBMode = "Explicit"
  lon_value = (/-60,0,60,120,180,240,300/)
  name = (/"W","","E","E","E","W","W"/)
  lon_lable = new((/7/),"string")
  lon_lable(0:4) = abs(lon_value(0:4))+"~S~o~N~"+name(0:4)
  lon_lable(5:6) = (lon_value(5:6)-120)+"~S~o~N~"+name(5:6)


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
  res@lbLabelBarOn     = True
  res@lbOrientation = "Horizontal"
  res@lbLabelFontHeightF = zt
  res@pmLabelBarWidthF = 0.85
  res@pmLabelBarHeightF = 0.06
  res@lbLabelStride = 1
  res@pmLabelBarOrthogonalPosF = 0.1
  res@lbFigureStringsFontHeightF = zt 
  res@lbBoxLinesOn = True
  res@lbBoxLineColor = "black"
  res@lbBoxLineThicknessF = bt

  res@gsnLeftString = "(a) Clim.   Nov.  300HGT&U"
  res@gsnLeftStringFontHeightF    = zt+0.002
  base1 = gsn_csm_contour_map(wks,cz1(10,:,:),res)
  ;------uclim
  cres1@cnLevelSelectionMode  = "ExplicitLevels" 
  cres1@cnLevels  = (/20,30,40,50,60,70/)
  cres1@cnLineThicknessF      = 4   
  cres1@cnLineColor = "green4";gray29" 
  cres1@gsnContourZeroLineThicknessF = 0. 
  cres1@cnLineLabelsOn        = True  
  cres1@cnLineLabelFontHeightF = 0.006
  cres1@cnLineDashPattern     = 0       
  cres1@cnInfoLabelOn         = False 
  cres1@cnFillOn              = False      
  cres1@cnLinesOn             = True
  cres1@gsnContourPosLineDashPattern = 0
  cres1@gsnContourNegLineDashPattern = 1

  plot1 = gsn_csm_contour(wks,cu1(10,:,:),cres1)
  overlay(base1, plot1)
draw(base1)


;--------------------------------------------------------plot2 
  re                   = True   
  re@gsnAddCyclic      = True   
  re@gsnDraw           = False        
  re@gsnFrame          = False        
  re@gsnLeftString     = ""
  re@gsnRightString    = ""
  re@cnFillOn              = True      
  re@cnLinesOn             = False
  re@cnFillPalette     = "MPL_bwr"
  ; 设置经纬度范围                 
  re@vpXF             = 0.11                 
  re@vpYF             = 0.68                 
  re@vpWidthF         = 0.85
  re@vpHeightF        = 0.35 

  re@mpDataSetName = "Earth..4"
  re@mpGeophysicalLineColor = "gray60";"gray30"
  re@mpGeophysicalLineThicknessF = 2
  re@tmBorderThicknessF = bt
  re@tmXBMajorThicknessF = bt
  re@tmYLMajorThicknessF = bt
  re@tmXBMinorThicknessF = bt
  re@tmYLMinorThicknessF = bt

  lonR = 180
  lonL = -180
  latN = 80
  latS = 0

  re@mpMaxLatF = latN
  re@mpMinLatF = latS
  ; re@mpMaxLonF = lonR
  ; re@mpMinLonF = lonL
  re@mpCenterLonF = 120
  
  re@tmXBMode = "Explicit"
  lon_value = (/-60,0,60,120,180,240,300/)
  name = (/"W","","E","E","E","W","W"/)
  lon_lable = new((/7/),"string")
  lon_lable(0:4) = abs(lon_value(0:4))+"~S~o~N~"+name(0:4)
  lon_lable(5:6) = (lon_value(5:6)-120)+"~S~o~N~"+name(5:6)


  lat_value1 = ispan(latS,latN,20)
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
  ;------------------------------------
  re@lbLabelBarOn     = True
  re@lbOrientation = "Horizontal"
  re@lbLabelFontHeightF = zt
  re@pmLabelBarWidthF = 0.85
  re@pmLabelBarHeightF = 0.06
  re@lbLabelStride = 1
  re@pmLabelBarOrthogonalPosF = 0.1
  re@lbFigureStringsFontHeightF = zt 
  re@lbBoxLinesOn = True
  re@lbBoxLineColor = "black"
  re@lbBoxLineThicknessF = bt

  re@gsnLeftString = "(b) Clim.   Nov.  500HGT"
  re@gsnLeftStringFontHeightF    = zt+0.002
  base2 = gsn_csm_contour_map(wks,cz2(10,:,:),re)
draw(base2)



;--------------------------------------------------------plot3
  re1                   = True   
  re1@gsnAddCyclic      = True   
  re1@gsnDraw           = False        
  re1@gsnFrame          = False        
  re1@gsnLeftString     = ""
  re1@gsnRightString    = ""
  re1@cnFillOn              = True      
  re1@cnLinesOn             = False
  re1@cnFillPalette     = "MPL_bwr"
  resv = re1
                 
  re1@vpXF             = 0.11                 
  re1@vpYF             = 0.38                
  re1@vpWidthF         = 0.85
  re1@vpHeightF        = 0.35 

  re1@mpDataSetName = "Earth..4"
  re1@mpGeophysicalLineColor = "gray60";"gray30"
  re1@mpGeophysicalLineThicknessF = 2
  re1@tmBorderThicknessF = bt
  re1@tmXBMajorThicknessF = bt
  re1@tmYLMajorThicknessF = bt
  re1@tmXBMinorThicknessF = bt
  re1@tmYLMinorThicknessF = bt

  lonR = 180
  lonL = -180
  latN = 80
  latS = 0

  re1@mpMaxLatF = latN
  re1@mpMinLatF = latS
  ; re1@mpMaxLonF = lonR
  ; re1@mpMinLonF = lonL
  re1@mpCenterLonF = 120

  re1@tmXBMode = "Explicit"
  lon_value = (/-60,0,60,120,180,240,300/)
  name = (/"W","","E","E","E","W","W"/)
  lon_lable = new((/7/),"string")
  lon_lable(0:4) = abs(lon_value(0:4))+"~S~o~N~"+name(0:4)
  lon_lable(5:6) = (lon_value(5:6)-120)+"~S~o~N~"+name(5:6)

  lat_value11 = ispan(latS,latN,20)
  lat_lable11 = abs(lat_value11)+"~S~o~N~N"
  re1@tmYLValues = lat_value11
  re1@tmYLLabels = lat_lable11
  re1@tmXBLabelFontHeightF = zt
  re1@tmYLMode = "Explicit"
  re1@tmYLLabelFontHeightF = zt
  re1@gsnLeftStringFontHeightF = zt
  re1@tmXBValues = lon_value
  re1@tmXBLabels = lon_lable


  re1@tmXBMajorLengthF = 0.006
  re1@tmXBMinorLengthF = 0.004
  re1@tmXBMajorOutwardLengthF = 0.006
  re1@tmXBMinorOutwardLengthF = 0.004
  re1@tmYLMajorLengthF = 0.006
  re1@tmYLMinorLengthF = 0.004
  re1@tmYLMajorOutwardLengthF = 0.006
  re1@tmYLMinorOutwardLengthF = 0.004
  re1@tmXBLabelDeltaF = -0.8
  re1@tmYLLabelDeltaF = -0.8
  
  re1@lbLabelFont   = "times-roman"
  re1@tmXBLabelFont = "times-roman" 
  re1@tmYLLabelFont = "times-roman"
  re1@tiXAxisFont   = "times-roman"
  re1@tiYAxisFont   = "times-roman"
  re1@tiMainFont    = "times-roman"
  re1@gsnStringFont = "times-roman"
  re1@cnInfoLabelOn = False       
  
  re1@lbLabelBarOn  = True
  re1@lbOrientation = "Horizontal"
  re1@lbLabelFontHeightF = zt
  re1@pmLabelBarWidthF = 0.85
  re1@pmLabelBarHeightF = 0.06
  re1@lbLabelStride = 1
  re1@pmLabelBarOrthogonalPosF = 0.1
  re1@lbFigureStringsFontHeightF = zt 
  re1@lbBoxLinesOn = True
  re1@lbBoxLineColor = "black"
  re1@lbBoxLineThicknessF = bt

  
  re1@gsnLeftString = "(c) Clim.   Nov.  SLP&UV"
  re1@gsnLeftStringFontHeightF    = zt+0.002
  base3 = gsn_csm_contour_map(wks,cslp(10,:,:),re1)



  ;-------------------------------------
  refmag = 5 
  resv@vcGlyphStyle         = "Fillarrow"  
  resv@vcFillArrowEdgeColor = "gray20";"black"
  resv@vcFillArrowEdgeThicknessF = 1
  resv@vcFillArrowFillColor = "gray20"
  resv@vcFillArrowWidthF = 0.0005   
  resv@vcFillArrowHeadYF = 0.15   
  resv@vcFillArrowHeadXF = 0.3
  resv@vcFillArrowHeadInteriorXF = 0.015 
  resv@vcMinDistanceF  = 0.015    
  ; resv@vcMinMagnitudeF = 0.2;0.6  

  resv@vcRefAnnoOn               = True  
  resv@vcRefMagnitudeF           = refmag 
  resv@vcRefLengthF              = 0.025 
  resv@vcRefAnnoBackgroundColor  = "white"      
  resv@vcRefAnnoPerimOn          = False       
  resv@vcRefAnnoFontHeightF      = 0.01        
  resv@vcRefAnnoString1On     = False           
  resv@vcRefAnnoString2On     = True    
  resv@vcRefAnnoString2       = refmag+"m/s"  
  resv@vcRefAnnoFont = "times-roman"
  resv@vcRefAnnoSide            = "Top" 
  resv@vcRefAnnoOrthogonalPosF  = -1.14 ;-0.435; 
  resv@vcRefAnnoParallelPosF    = 0.998  

  plot3 = gsn_csm_vector(wks,cu2(10,:,:),cv2(10,:,:),resv)
  overlay(base3, plot3)

draw(base3)
frame(wks)
end





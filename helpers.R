# =================#
# Helper functions #
# =================#

get_GPP <- function(clim = s1, params){
  
  output <- PRELES(PAR = clim$PAR, TAir = clim$TAir, VPD = clim$VPD, Precip = clim$Precip, CO2 = clim$CO2, fAPAR = clim$fAPAR, p = params)
  
  return(output)
  
}
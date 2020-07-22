library(tidyverse)
library(dplyr)
library(ProfoundData)
library(zoo)
library(lubridate)

# Download PROFOUND sqlite database to specified location and connect to database:
#   downloadDatabase(location = "data")
#   unzip("data/ProfoundData.zip")
vignette("ProfoundData")
setDB("ProfoundData.sqlite")
getDB()

# Explore the database
overview <- browseData()
# Use only stands with all required data sets available and pick a site. 
# Hyytiala
sites <- overview$site[!unlist(apply(overview[,c("CLIMATE_LOCAL", "FLUX", "METEOROLOGICAL", "MODIS", "SOILTS")], 1,  function(x) any(0 %in% x)))]

# Define a period (see Elias Schneider)
period = c("2001-01-01", "2008-12-31")

#================#
# Input variables#
#================#

# Generate the climatic variables that will be used for PRELES input. These are
#   PAR: daily sums of photosynthetically active radiation, mmol/m2.
#   TAir: daily mean temperature, degrees C.
#   VPD: daily mean vapour pressure deficits
#   PRECIP: daily rainfall, mm
#   CO2: air CO2
#   fAPAR: fractions of absorbed PAR by the canopy, 0-1 unitless

# We will need the data sets:
#   CLIMATE_LOCAL, FLUX, METEOROLOGICAL, MODIS ( SOIL_TS)
#   

#VPD
VPD_fun <- function(temperature, rel_hum){
  
  # this function takes the temperature in celsius degree and the relative humidity in percent and returns the vapor pressure deficit.
  
  # convert temperature from degree celsius to rankine
  temperature <- temperature*9/5+491.67
  
  # compute the saturation pressure (https://en.wikipedia.org/wiki/Vapour-pressure_deficit)
  vp_sat <- exp(-1.044*10^4/temperature-11.29-2.7*10^(-2)*temperature+1.289*10^(-5)*temperature^2-2.478*10^(-9)*temperature^3 + 6.456*log(temperature))
  
  # compute vapor pressure deficit
  VPD <- vp_sat*(1-rel_hum/100)
  
  return(VPD)
}


get_profound_input <- function(period, site, VPDcalc="manual"){

  clim_local <- getData("CLIMATE_LOCAL", site=site, period = period)
  TAir <- clim_local$tmean_degC
  PRECIP <- clim_local$p_mm
  
  # PAR
  #Transform units - Radiation: PAR https://en.wikipedia.org/wiki/Photosynthetically_active_radiation (Thanks to Elisa Schneider)
  joule2mol <- function(rad_Jcm2day) {((rad_Jcm2day* (2.2*(10^(-7))) / (299792458 * (6.626070150 * (10 ^(-34)))))/(6.02*(10^23))) / 0.0001}
  PAR <- joule2mol(clim_local$rad_Jcm2day)

  if(VPDcalc=="manual"){
    VPD <- VPD_fun(temperature = TAir, rel_hum = clim_local$relhum_percent)
  }else{
    meteo <- getData("METEOROLOGICAL", site = site, period = period)
    VPD <- meteo %>% 
      group_by(date(date)) %>%
      summarise(VPD = mean(vpdFMDS_hPa)/10) %>% # convert from hPA to kPa
      select(VPD)
  }
  
  #CO2
  # CO2 on a yearly basis.
  # NOTE: historical (and low) CO2 values result in negative GPP!!
  #CO2_isimip <- getData("CO2_ISIMIP", site=site, period = period)
  #CO2_isimip <- CO2_isimip[CO2_isimip$year %in% unique(clim_local$year),]
  #CO2_isimip <- do.call(rbind, lapply(split(CO2_isimip, as.factor(CO2_isimip$year)), function(x) x[sample(nrow(x), 1),]))
  #CO2 <- merge(CO2_isimip, clim_local, by = "year")$co2_ppm
  
  # Make CO2 constant.
  CO2 <- rep(380, times=nrow(clim_local))

  # fAPAR (thanks to Elias Schneider)
  # assumes the same fAPAR values throughout a period of 8 days.
  modis <- getData(dataset = "MODIS", site = site, period = period)
  schaltjahre <- clim_local %>% group_by(year) %>% summarise(len = length(day)) %>% select(len) %>% mutate(len = ifelse(len==366, 6, 5))
  d <- as.numeric(difftime(c(as.Date(modis$date[-1]), as.Date(modis$date[length(modis$date)])+schaltjahre$len[nrow(schaltjahre)]), as.Date(modis$date))) # why 5?
  # approximate missing values with na.approx from package zoo.
  fAPAR <- na.approx(rep(modis$fpar_percent, times=d))

  df <- data.frame(PAR=PAR, TAir=TAir, VPD=VPD, Precip=PRECIP, CO2=CO2, fAPAR=fAPAR, date = as.character(clim_local$date) , DOY=clim_local$day, site=as.character(site))
  
  return(df)
}

X <- get_profound_input(period=period, site=sites[1], VPDcalc = "manual")

for(i in 2:length(sites)){
  X <- rbind(X, get_profound_input(period=period, site=sites[i], VPDcalc = F))
}


#==================#
# Output variables #
#==================#

# Generate the climatic variables that will alo be PRELES output. These are
#   GPP: Gross Primary Production
#   ET: Evapotranspiration
#   SwC: Soil water balance.

get_profound_output <- function(period, site, vars = c("GPP")){
  

  #GPP
  flux <-  getData("FLUX", site=site, period = period)
  GPP <- flux %>% 
    group_by(lubridate::date(date)) %>% 
    summarise(GPP = mean(gppDtVutRef_umolCO2m2s1)) %>% 
    select(GPP)
  df <- data.frame(GPP=GPP)

  
  if("SW" %in% vars){
    # soil water balance
    soil_ts <-  getData("SOILTS", site=site, period = period)
    #   (or use porosity_percent from SOIL data?)
    SW <- soil_ts %>% 
      group_by(date(date)) %>% 
      summarise(SW = mean(swcFMDS1_degC)) %>% 
      select(SW)
    df$SW <- SW
  }
  # evapotranspiration
  
  return(df)

}

y <- get_profound_output(period=period, site=sites[1], vars=c("GPP"))
for(i in 2:length(sites)){
  y <- rbind(y, get_profound_output(period=period, site=sites[i], vars=c("GPP")))
}

save(y, file="Rdata/profound/profound_out.Rdata")
write.table(y, file="data/profound/profound_out", sep = ";",row.names = FALSE)

save(X, file="Rdata/profound/profound_in.Rdata")
write.table(X, file="data/profound/profound_in", sep = ";", row.names = FALSE)

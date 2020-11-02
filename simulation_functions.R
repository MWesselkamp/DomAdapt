require(mgcv)
require(mvtnorm)
library(Rpreles)
library(lhs)
library(tidyverse)
library(dplyr)
library(msm)

# helper functions
source("utils.R")

set.seed(123)

#==================#
# CLIMATE SIMULATOR#
#==================#

climate_simulator = function(seq_len, sample, zero=TRUE){

  year = sample(2001:2008, 1)
  doy = sample(1:(365-seq_len), 1)
  
  T_hat = predict(fmT, data.frame(DOY = doy:(doy+seq_len-1), year=year))
  PAR_hat = predict(fmPAR, data.frame(DOY = doy:(doy+seq_len-1), year=year))
  VPD_hat = predict(fmVPD, data.frame(DOY = doy:(doy+seq_len-1), year=year))
  Precip_hat = predict(fmPrecip, data.frame(DOY = doy:(doy+seq_len-1), year=year))
  fAPAR_hat = predict(fmfAPAR, data.frame(DOY = doy:(doy+seq_len-1), year=year))
  
  res_mat = data.frame(TAir=fmT$residuals, 
                      PAR=fmPAR$residuals, 
                      VPD=fmVPD$residuals, 
                      Precip=fmPrecip$residuals, 
                      fAPAR = fmfAPAR$residuals)
  
  cov_mat = cov(res_mat)
  
  noise = rmvnorm(seq_len, mean=rep(0, length=ncol(res_mat)), sigma=cov_mat)
  
  T_hat = T_hat + noise[,1]
  PAR_hat = PAR_hat + noise[,2]
  VPD_hat = VPD_hat + noise[,3]
  Precip_hat = Precip_hat + noise[,4]
  fAPAR_hat = fAPAR_hat + noise[,5]
  
  if(zero){
    PAR_hat[which(PAR_hat<0)] = 0
    VPD_hat[which(VPD_hat<0)] = 0
    Precip_hat[which(Precip_hat<0)] = 0
    fAPAR_hat[which(fAPAR_hat<0)] = 0
  }
  
  climsim = data.frame(TAir = T_hat,
                      PAR = PAR_hat,
                      VPD = VPD_hat,
                      Precip = Precip_hat,
                      fAPAR = fAPAR_hat,
                      CO2 = 380,
                      DOY = doy:(doy+seq_len-1),
                      year = year,
                      sample = sample)
  
  return(climsim)
  
}


#=====================#
# Parameter Sampling  #
#=====================#


get_parameters <- function(){
  
  # calibrated paramters (Elisa Schneider)
  load("Rdata/OptimizedParametersPreles.Rdata")

  # Parameter default values directly taken from Rpreles GitHub repository. Merge with ES values.
  pars <- read.csv2("data/parameter_default_values.csv")
  
  pars <- pars %>% 
    mutate(Min = par$min[1:30],Max = par$max[1:30]) %>% 
    rename(Name = X, Default = Value)
  
  return(pars)
}


# select a range of parameters for sampling (influential model parameters, taken from Minunno, Plein and Schneider).
# sample params in Latin Hypercube design
sample_parameters <- function(pars, samples, normal, pars_names = c("beta", "X0", "gamma", "alpha", "chi")){
  
  pars_influential <- pars %>% 
    filter(Name %in% pars_names) %>% 
    mutate(std = abs(Max-Min)/4)
  
  if (normal){
    lhs <- randomLHS(samples, nrow(pars_influential))
    for (i in 1:length(pars_influential)){
      lhs[,i] = qtnorm(lhs[,i], mean = pars_influential$Default[i], sd=pars_influential$std[i], 
                       lower=pars_influential$Min[i], upper = pars_influential$Max[i])
    }
    pars_lhs = t(lhs)
      
  }else{
    lhs <- randomLHS(samples, nrow(pars_influential))
    # Generate stratified parameter combinations by mapping lhs to data space.
    pars_lhs <- apply(lhs, 1, function(x) pars_influential$Min + x*abs(pars_influential$Max-pars_influential$Min))
  }

  return(pars_lhs)
  
}

#========================#
# Simulating from PRELES #
#========================#

# a function that passes the input data (climate and parameters) to the RPreles function.
get_preles_output <- function(clim, params, returncols){
  
  # the function takes:
  #   Climate data containing the variables PAR, TAir, VPS, Precip, CO2, fAPAR
  # it eturns:
  #   Preles output, containing variables GPP, ET, SW.
  
  output <- PRELES(TAir = clim$TAir, PAR = clim$PAR, VPD = clim$VPD, Precip = clim$Precip, fAPAR = clim$fAPAR, CO2 = clim$CO2,  p = params, returncols = returncols)
  
  return(output)
  
}

#===========================#
# Latin Hypercube Sampling  #
#===========================#

# set seed for reproducability
set.seed(123)

# load packages
library(Rpreles)
library(lhs)
library(tidyverse)
library(dplyr)

# get helper functions
source("utils.R")


get_parameters <- function(default=TRUE){
  
  if(default){
    #   Default parameter values and the minimum and maximum ranges used for sensitivity analysis in Masterthesis of Elisa Schneider.
    load("Rdata/ParameterRangesPreles.Rdata")# par
  }else{
    # calibrated paramters (Elisa Schneider)
    load("Rdata/OptimizedParametersPreles.Rdata")
  }
  # Parameter default values directly taken from Rpreles GitHub repository. Merge with ES values.
  pars <- read.csv2("data/parameter_default_values.csv")

  pars <- pars %>% 
    mutate(Min = par$min[1:30],Max = par$max[1:30]) %>% 
    rename(Name = X, Default = Value)
  
  return(pars)
}


# select a range of parameters for sampling (influential model parameters, taken from Minunno, Plein and Schneider).
# sample params in Latin Hypercube design
sample_parameters <- function(pars_default, nsamples, LHS = "random", pars_names = c("beta", "X0", "gamma", "alpha", "chi") ){
  
  pars_influential <- pars_default %>% 
    filter(Name %in% pars_names)
  
  # Create useful variable
  npars <- nrow(pars_influential) # number of influential parameters
  
  # generate LHS values
  if(LHS == "random"){ # uniformly distributed
  lhs <- randomLHS(nsamples, npars)
  }
  
  # Generate stratified parameter combinations by mapping lhs to data space.
  pars_lhs <- t(apply(lhs, 1, function(x) pars_influential$Min + x*abs(pars_influential$Max-pars_influential$Min)))
  
  dimnames(pars_lhs) <- list(NULL, pars_names)
  
  return(pars_lhs)
  
}


# a function that passes the input data (climate and parameters) to the RPreles function.
get_preles_output <- function(clim, params, return_cols){
  
  # the function takes:
  #   Climate data containing the variables PAR, TAir, VPS, Precip, CO2, fAPAR
  # it eturns:
  #   Preles output, containing variables GPP, ET, SW.
  
  output <- PRELES(TAir = clim[,1], PAR = clim[,2], VPD = clim[,3], Precip = clim[,4], fAPAR = clim[,5], CO2 = clim[,6],  p = params, returncols = return_cols)
  
  return(output)
  
}

# Generate GPP data from stratified parameter combinations.
get_simulations <- function(pars, clim_data, data_dir, pars_lhs = parsLHS, vars=c("GPP", "SW")){
  
  # This function generates and saves GPP, EV and SW data from climate and parameter lhs input.
  
  # Create useful variable
  par_names <- dimnames(pars_lhs)[[2]]
  inf_ind <- which(as.character(pars$Name) %in% par_names) # indices of influential parameters.
  nsamples = dim(clim_data)[1]
  days = dim(clim_data)[2]
  var_out = length(vars)
  
  pars <- pars$Default
  
  outputs = vector(mode="list", length=nsamples*nsamples)
  mat <- matrix(pars, nrow = length(pars), ncol = nsamples)
  mat[inf_ind,] <- t(pars_lhs)
  
  for(i in 1:nsamples){
    clim <- clim_data[i,,]
    for (j in 1:nsamples) {
      pars <- mat[,j]
      
      output <- get_preles_output(clim = clim, params = pars, return_cols = vars)
      
      # write output (GPP, SW)
      write.table(matrix(unlist(output), nrow = days, ncol=var_out, dimnames = list(NULL, vars)),
                  file = paste0(data_dir, "sim",i,"_",j, "_out"), row.names = F, sep=";")
      # write input (Climate data + Parameters)
      write.table(cbind(clim, matrix(rep(pars[inf_ind], each=days), ncol = ncol(parsLHS), nrow=days, dimnames = list(NULL, par_names))),
                  file = paste0(data_dir, "sim",i,"_",j, "_in"), row.names = F, sep=";")
    }
  }
  #return(outputs)
}
#===========================#
# Latin Hypercube Sampling  #
#===========================#

# set seed for reproducability
set.seed(123)

# if not available,install packages
if(!require(Rpreles)){devtools::install_github('MikkoPeltoniemi/Rpreles')}
if(!require(lhs)){install.packages("lhs")}
if(!require(abind)){install.packages("abind")}

# load packages
library(Rpreles)
library(lhs)
library(tidyverse)
library(dplyr)

# get helper functions
source("utils.R")

# Load data descending from the Master Thesis of Elias Schneider (based on Minunno et al. (2016)):
#   Climatic input for four boreal sites in Finland.
load("Rdata/borealsites/EddyCovarianceDataBorealSites.RData")
load("Rdata/profound/profound_input.RData")

# choose the data set to use (X or s1-4)
climate_data <- X # profound input
samples <- 100
data_dir <- "data/preles/exp/"

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
sample_parameters <- function(pars_default, nsamples = samples, LHS = "random", pars_names = c("beta", "X0", "gamma", "alpha", "chi") ){
  
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

# Generate GPP data from stratified parameter combinations.
get_lhs_output <- function(nsamples = samples, pars_lhs = parsLHS, pars = pars_def, clim=climate_data, vars=c("GPP", "SW")){
  
  # This function generates and saves GPP, EV and SW data from climate and parameter lhs input.
  
  # Create useful variable
  par_names <- dimnames(pars_lhs)[[2]]
  inf_ind <- which(as.character(pars$Name) %in% par_names) # indices of influential parameters.
  
  pars <- pars$Default
  
  mat <- matrix(pars, nrow = length(pars), ncol = nsamples)
  mat[inf_ind,] <- t(pars_lhs)
  
  output <- apply(mat, 2, function(y) get_preles_output(clim = clim, params = y, return_cols = vars))
  
  var_out = length(vars)
  
  for(i in 1:length(output)){
    
    write.table(matrix(unlist(output[[i]]), nrow = nrow(clim), ncol=var_out, dimnames = list(NULL, vars)),
               file = paste0(data_dir, "sim",i, "_out"), row.names = F, sep=";")
    
    
  }
  
  return(output)
  
}

write_input_data <- function(clim=climate_data, pars_lhs = parsLHS){
  
  len <-  nrow(clim)
  pars_names <- dimnames(parsLHS)[[2]]
  
  for(i in 1:nrow(pars_lhs)){
    write.table(cbind(clim, matrix(rep(parsLHS[1,], each=len), ncol = ncol(parsLHS), nrow=len, dimnames = list(NULL, pars_names))),
             file = paste0(data_dir, "sim",i, "_in"), row.names = F, sep=";")
  }
}


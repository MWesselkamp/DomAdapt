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
load("data/borealsites/EddyCovarianceDataBorealSites.RData")


get_default_parameters <- function(){
  
  #   Default parameter values and the minimum and maximum ranges used for sensitivity analysis in Masterthesis of Elisa Schneider.
  load("data/ParameterRangesPreles.Rdata") # par
  # Parameter default values directly taken from Rpreles GitHub repository. Merge with ES values.
  pars_default <- read.csv2("data/parameter_default_values.csv")

  pars_default <- pars_default %>% 
    mutate(Min = par$min[1:30],Max = par$max[1:30]) %>% 
    rename(Name = X, Default = Value)
  
  return(pars_default)
}

pars_def <- get_default_parameters()

# Run model exemplarly with default parameters at stand s1.
o1 <- get_preles_output(clim=s1, params=pars_def$Default) # Works
# Plot model predictions and s1 observations.
plot_preles_output(output = o1, nsamples = 1, all = F)


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

parsLHS <- sample_parameters(pars_default = pars_def, nsamples = 20)

# Generate GPP data from stratified parameter combinations.
get_lhs_output <- function(nsamples = 20, pars_lhs = parsLHS, pars = pars_def, clim = s1){
  
  # This function generates and saves GPP, EV and SW data from climate and parameter lhs input.
  
  # Create useful variable
  par_names <- dimnames(pars_lhs)[[2]]
  inf_ind <- which(as.character(pars$Name) %in% par_names) # indices of influential parameters.
  
  pars <- pars$Default
  
  mat <- matrix(pars, nrow = length(pars), ncol = nsamples)
  mat[inf_ind,] <- t(pars_lhs)
  
  output <- apply(mat, 2, function(y) get_preles_output(clim = clim, params = y))
  
  var_out = 3
  var_names = c("GPP", "ET", "SW")
  
  for(i in 1:length(output)){
    
    write.table(matrix(unlist(output[[i]]), nrow = nrow(s1), ncol=var_out, dimnames = list(NULL, var_names)),
               file = paste0("data/preles/sim",i, "_out"), row.names = F)
    
    
  }
  
}

get_lhs_output()

write_input_data <- function(clim, pars_lhs = parsLHS){
  
  len <-  nrow(clim)
  pars_names <- dimnames(parsLHS)[[2]]
  
  for(i in 1:nrow(pars_lhs)){
    write.table(cbind(clim, matrix(rep(parsLHS[1,], each=len), ncol = ncol(parsLHS), nrow=len, dimnames = list(NULL, pars_names))),
             file = paste0("data/preles/sim",i, "_in"), row.names = F)
  }
}

write_input_data(clim=s1)


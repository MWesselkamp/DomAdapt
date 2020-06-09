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
library(abind)

# get helper functions
source("helpers.R")

# Load data descending from the Master Thesis of Elias Schneider (based on Minunno et al. (2016)):
#   Climatic input for four boreal sites in Finland.
#   Default parameter values and the minimum and maximum ranges used for sensitivity analysis.

load("data/EddyCovarianceDataBorealSites.RData")
load("data/ParameterRangesPreles.Rdata")

# Parameter default values directly taken from Rpreles GitHub repository. Merge with ES values.
pars_default <- read.csv2("data/parameter_default_values.csv")

pars_default <- pars_default %>% 
  mutate(Min = par$min[1:30],Max = par$max[1:30]) %>% 
  rename(Name = X, Default = Value)

# Run model exemplarly with default parameters at stand s1.
o1 <- get_output(clim=s1, params=pars_default$Default) # Works
#reshape to array
arr <- array(unlist(o1), dim = c(length(o1[[1]]),length(o1),  1))


# Plot model predictions and s1 observations.
par(mfrow=c(3,1))
plot(o1$GPP, type = "l", col="darkgreen")
lines(s1$GPPobs, type="l", col="green", lwd=0.5)
plot(o1$ET, type="l", col="red")
lines(s1$ETobs, type="l", col="darkred")
plot(o1$SW, type="l", col="blue") 

# select a range of parameters for sampling (influential model parameters, taken from Minunno, Plein and Schneider).
pars_names <- c("beta", "X0", "gamma", "alpha", "chi") # so far, these have to be in the same order.
pars_influential <- pars_default %>% 
  filter(Name %in% pars_names)

# Create useful variables.
inf_ind <- which(pars_default$Name %in% pars_influential$Name) # indices of influential parameters.
nsamples <- 20 # number of stratified samples
npars <- nrow(pars_influential) # number of influential parameters
nout <- 3 # number of output variables.
ndays <- nrow(s1)

# generate uniformly distributed LHS
lhs <- randomLHS(nsamples, npars)
# Generate stratified parameter combinations by mapping lhs to data space.
pars_lhs <- t(apply(lhs, 1, function(x) pars_influential$Min + x*abs(pars_influential$Max-pars_influential$Min)))

# Create input array 
e <- apply(pars_lhs, 1, function(x) rep(x, times = ndays))
pars_arr <- array(e, dim = c(npars, ndays, nsamples))
clim_arr <- array(rep(t(as.matrix(s1)), times=nsamples), dim = c(length(s1), ndays, nsamples))

input_arr <- abind(clim_arr, pars_arr, along = 1)
save(input_arr, file="Rdata/input_arr.Rdata")

# Generate GPP data from stratified parameter combinations.

# replace default parameter values
pars <- pars_default$Default
mat <- matrix(pars, nrow = length(pars), ncol = nsamples)
mat[inf_ind,] <- t(pars_lhs)

output <- apply(mat, 2, function(y) get_GPP(params = y))
plot_output(output, all = TRUE)

# reshape to array
output_arr <- array(unlist(output), dim = c(nrow(s1), nout,  nsamples))
save(output_arr, file="Rdata/output_arr.Rdata")


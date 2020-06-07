#===========================#
# Latin Hypercube Sampling  #
#===========================#

# set seed for reproducability
set.seed(123)

# if not available,install packages
if(!require(Rpreles)){devtools::install_github('MikkoPeltoniemi/Rpreles')}
if(!require(lhs)){install.packages("lhs")}

# load packages
library(Rpreles)
library(lhs)
library(tidyverse)
library(dplyr)

# get helper functions
source("helpers.R")

# Load data descending from the Master Thesis of Elias Schneider (based on Minunno et al. (2016)):
#   Climatic input for four boreal sites in Finland.
#   Default parameter values and the minimum and maximum ranges used for sensitivity analysis.

load("data/EddyCovarianceDataBorealSites.RData")
load("data/ParameterRangesPreles.Rdata")

# Parameter default values directly taken from Rpreles GitHub repository. Merge with ES values.
pars_default <- t(data.frame(
  ## SITE AND SOIL RELATED
  soildepth = 413.0, ## 1 soildepth
  ThetaFC = 0.450, ## 2 ThetaFC
  ThetaPWP = 0.118, ## 3 ThetaPWP
  tauDrainage = 3 , ## 4 tauDrainage
  ## GPP_MODEL_PARAMETERS
  beta = 0.748018, ## 5 betaGPP
  tau = 13.23383, ## 6 tauGPP
  X0 = -3.9657867, ## 7 S0GPP
  Smax = 18.76696, ## 8 SmaxGPP
  kappa = -0.130473, ## 9 kappaGPP
  gamma = 0.034459, ## 10 gammaGPP
  rhoP = 0.450828, ## 11 soilthresGPP
  cmCO2 = 2000, ## 12 cmCO2
  ckappaCO2 = 0.4, ## 13 ckappaCO2
  ## EVAPOTRANSPIRATION_PARAMETERS
  alpha = 0.324463, ## 14 betaET
  lambda = 0.874151, ## 15 kappaET
  chi = 0.075601, ## 16 chiET
  rhoET = 0.541605, ## 17 soilthresET
  nu = 0.273584, ## 18 nu ET
  ## SNOW_RAIN_PARAMETERS
  Metlcoeff = 1.2, ## 19 Meltcoef
  I_0 = 0.33, ## 20 I_0
  CWmax = 4.970496, ## 21 CWmax, i.e. max canopy water
  SnowThreshold = 0, ## 22 SnowThreshold, 
  T_0 = 0, ## 23 T_0, 
  ## Initialisation
  SWinit = 200, ## 24 SWinit, ## START INITIALISATION PARAMETERS 
  CWinit = 0, ## 25 CWinit, ## Canopy water
  SOGinit = 0, ## 26 SOGinit, ## Snow on Ground 
  Sinit = 20, ## 27 Sinit ##CWmax
  t0 = -999, ## t0 fPheno_start_date_Tsum_accumulation; conif -999, for birch 57
  tcrit =  -999, ## tcrit, fPheno_start_date_Tsum_Tthreshold, 1.5 birch
  tsumcrit = -999 ##tsumcrit, fPheno_budburst_Tsum, 134 birch
))

pars_default = data.frame(
  Name=row.names(pars_default),
  Default = pars_default,
  Min = par$min[1:30],
  Max = par$max[1:30])

# Run model exemplarly with default parameters at stand s1.
o1 <- PRELES(PAR = s1$PAR, TAir = s1$TAir, VPD = s1$VPD, Precip = s1$Precip, CO2 = s1$CO2, fAPAR = s1$fAPAR,p = pars_default$Default)

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

# indices of influential parameters.
inf_ind <- which(pars_default$Name %in% pars_influential$Name)
nsamples <- 20
npars <- nrow(pars_influential)

# generate uniformly distributed LHS
lhs <- randomLHS(nsamples, npars)
# Generate stratified parameter combinations by mapping lhs to data space.
pars_lhs <- t(apply(lhs, 1, function(x) pars_influential$Min + x*abs(pars_influential$Max-pars_influential$Min)))

# Generate GPP data from stratified parameter combinations.

# replace default parameter values
pars <- pars_default$Default
mat <- matrix(pars, nrow = length(pars), ncol = nsamples)
mat[inf_ind,] <- t(pars_lhs)

output <- as.data.frame(matrix(unlist(apply(mat, 2, function(y) get_GPP(params = y))), nrow=nsamples*nrow(s1), ncol = 3))

output <- output %>% 
  mutate(sim = rep(1:nsamples, each = nrow(s1)),
         DOY = rep(1:nrow(s1), times = nsamples))

ggplot(output) +
  geom_path(aes(x = DOY, y = V1, group = sim))

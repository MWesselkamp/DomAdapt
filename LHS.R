#=====================#
# Simulate GPP data   #
#=====================#

if(!require(Rpreles)){devtools::install_github('MikkoPeltoniemi/Rpreles')}
library(Rpreles)

# Load data descending from the Master Thesis of Elias Schneider (based on Minunno et al. (2016)):
#   Climatic input for four boreal sites in Finland.
#   Default parameter values and the minimum and maximum ranges used for sensitivity analysis.

load("data/EddyCovarianceDataBorealSites.RData")
load("data/ParameterRangesPreles.Rdata")

# Run model with this data at stand s1.
o1 <- PRELES(PAR = s1$PAR, TAir = s1$TAir, VPD = s1$VPD, Precip = s1$Precip, CO2 = s1$CO2, fAPAR = s1$fAPAR)

par(mfrow=c(3,1))
plot(o1$GPP, type = "l", col="darkgreen")
lines(s1$GPPobs, type="l", col="green", lwd=0.5)
plot(o1$ET, type="l", col="red")
lines(s1$ETobs, type="l", col="darkred")
plot(o1$SW, type="l", col="blue")


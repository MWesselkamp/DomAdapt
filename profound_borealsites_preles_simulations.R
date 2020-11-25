#===============================================#
# Simulating GPP vom PREles with available data #
#===============================================#

# load packages
library(Rpreles)
library(ggplot2)

source("simulation_functions.R")

params = get_parameters()


#================#
## Profound data #
#================#

load("Rdata/profound/profound_in.Rdata") # X
load("Rdata/profound/profound_out.Rdata") # y
#X <- X_test

output <- PRELES(TAir = X$TAir, PAR = X$PAR, VPD = X$VPD, Precip = X$Precip, fAPAR = X$fAPAR, CO2 = X$CO2,  p = params$Default, returncols = c("GPP", "SW", "ET"))

y_pred <- data.frame("GPP_pred" = output[[1]])
X_full = cbind(X, y)
X_full= cbind(X_full, y_pred)

ggplot(X_full) + 
  geom_point(aes(x=date, y=GPP, group=site, colour="Observations")) +
  geom_line(aes(x= date, y=GPP_pred, group=site, colour="Preles Predictions")) +
  facet_wrap(.~site) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), legend.position = "bottom") +
  scale_color_manual(name="", values=c("black", "red"))

y_preles = as.data.frame(do.call(cbind, output))
save(y_preles, file="Rdata/profound/preles_out.Rdata")
write.table(y_preles, file="data/profound/preles_out", sep = ";",row.names = FALSE)
#==============================#
#Profound data: Test Year 2008 #
#==============================#

for (site in c("hyyitala", "bily_kriz")){

  x = X[which(X$site=="hyytiala" & X$year == 2008),]

  load("Rdata/CalibratedParametersHytProf.Rdata")

  output2008calib <- PRELES(TAir = x$TAir, PAR = x$PAR, VPD = x$VPD, Precip = x$Precip, fAPAR = x$fAPAR, CO2 = x$CO2,  p = par$calib[1:30], returncols = c("GPP"))
  output2008def <- PRELES(TAir = x$TAir, PAR = x$PAR, VPD = x$VPD, Precip = x$Precip, fAPAR = x$fAPAR, CO2 = x$CO2,  p = pars$Value[1:30], returncols = c("GPP"))
  output2008calib = as.data.frame(output2008calib)
  output2008def = as.data.frame(output2008def)

  write.table(output2008calib, file=paste0("data/profound/output", site, "2008calib"), sep = ";",row.names = FALSE)
  write.table(output2008def, file=paste0("data/profound/output", site, "2008def"), sep = ";",row.names = FALSE)

}

#====================#
## Boreal Sites data #
#====================#

load("Rdata/borealsites/boreal_sites_in.Rdata") # X
load("Rdata/borealsites/boreal_sites_out.Rdata") # y

# approximate VPD for getting tht output. These rows will be removed in Python again.
boreal_sites_in$VPD <- zoo::na.approx(boreal_sites_in$VPD)

output <- PRELES(TAir = boreal_sites_in$TAir, PAR = boreal_sites_in$PAR, VPD = boreal_sites_in$VPD, Precip = boreal_sites_in$Precip, fAPAR = boreal_sites_in$fAPAR, CO2 = boreal_sites_in$CO2,  p = params$Default, returncols = c("GPP", "SW", "ET"))

y <- as.data.frame(do.call(cbind, output))
save(y, file="Rdata/borealsites/preles_out.Rdata")
write.table(y, file="data/borealsites/preles_out", sep = ";",row.names = FALSE)



#==========================#
# Test for autocorrelation #
#==========================#

# Use of Borealsites data.
load("Rdata/borealsites/EddyCovarianceDataBorealSites.RData")

output <- PRELES(TAir = s1$TAir, PAR = s1$PAR, VPD = s1$VPD, Precip = s1$Precip, fAPAR = s1$fAPAR, CO2 = s1$CO2,  p = params$Default, returncols = c("GPP", "SW", "ET"))

# 1. Fit a random forest to GPP simulations, dependent on borealsites data.

require(randomForest)
require(mgcv)

par(mfrow=c(1,2))
rf2 <- randomForest(output[[1]] ~ PAR + TAir + VPD + Precip + CO2 + fAPAR + DOY, data = s1)
rf_acf2 <- acf((rf2$predicted-output[[1]])^2, main="AC of RF Residuals (Boreal sites S1 ) \n DOY included")

rf <- randomForest(output[[1]] ~ PAR + TAir + VPD + Precip + CO2 + fAPAR + DOY, data = s1)
rf_acf <- pacf((rf$predicted-output[[1]])^2, main="PAC of RF Residuals (Boreal sites S1 ) \n DOY included")



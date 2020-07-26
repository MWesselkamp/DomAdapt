#===============================================#
# Simulating GPP vom PREles with available data #
#===============================================#

# load packages
library(Rpreles)
library(ggplot2)

#================#
## Profound data #
#================#

source("preles_simulations.R")

load("Rdata/profound/profound_in.Rdata") # X
load("Rdata/profound/profound_in_test.Rdata") # X_test
load("Rdata/profound/profound_in_trainval.Rdata") # X
load("Rdata/profound/profound_out.Rdata") # y

params = get_parameters(default = FALSE)
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
save(y_preles, file="Rdata/profound/preles_out_trainval.Rdata")
write.table(y_preles, file="data/profound/preles_out_trainval", sep = ";",row.names = FALSE)

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

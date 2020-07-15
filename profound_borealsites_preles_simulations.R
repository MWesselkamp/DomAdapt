#=======================================#
# Preles simulations with available data#
#=======================================#

# load packages
library(Rpreles)

## Profound data

source("preles_simulations.R")

load("Rdata/profound/profound_in.Rdata") # X
load("Rdata/profound/profound_out.Rdata") # y

params = get_parameters(default = FALSE)

output <- PRELES(TAir = X$TAir, PAR = X$PAR, VPD = X$VPD, Precip = X$Precip, fAPAR = X$fAPAR, CO2 = X$CO2,  p = params$Default, returncols = c("GPP", "SW", "ET"))

par(mfrow=c(3,1))
for(i in 1:3){
plot(output[[i]], type="l")
}

y <- do.call(cbind, output)
save(y, file="Rdata/profound/preles_out.Rdata")
write.table(y, file="data/profound/preles_out", sep = ";",row.names = FALSE)

## Boreal Sites data

load("Rdata/boreal_sites/boreal_sites_in.Rdata") # X
load("Rdata/boreal_sites/boreal_sites_out.Rdata") # y

output <- PRELES(TAir = boreal_sites_in$TAir, PAR = boreal_sites_in$PAR, VPD = boreal_sites_in$VPD, Precip = boreal_sites_in$Precip, fAPAR = boreal_sites_in$fAPAR, CO2 = boreal_sites_in$CO2,  p = params$Default, returncols = c("GPP", "SW", "ET"))

par(mfrow=c(3,1))
for(i in 1:3){
  plot(output[[i]], type="l")
}

y <- do.call(cbind, output)
save(y, file="Rdata/boreal_sites/preles_out.Rdata")
write.table(y, file="data/boreal_sites/preles_out", sep = ";",row.names = FALSE)

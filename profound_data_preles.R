
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

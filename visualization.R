#===============#
# Profound data #
#===============#

X <- read.csv2("data/profound/profound_input")
y <- read.csv2("data/profound/profound_output")
load("Rdata/borealsites/EddyCovarianceDataBorealSites.RData")

# Plot: Profound vs. Boreal sites
par(mfrow=c(2,3), xpd=F)
for(i in 1:6){
  plot(X[,i], type="l", col="blue", main = colnames(X[i]), ylab = "Value", xlab = "Day of year")
  lines(s1[,i], type="l", col="red")
  legend(x = "top",legend=c("BorealSites", "Profound"), col = c("red", "blue"), lty=1, lwd=1, horiz = T, xpd=T)

}

climate_data <- X
climate_data$CO2 <- s1$CO2

# Plot preles output for climate data X and default parameter values.
o1 <- get_preles_output(clim=climate_data, params=pars_def$Default, return_cols = c("GPP", "SW")) # Works
# Plot model predictions and s1 observations.
plot_preles_output(output = o1, nsamples = 1, all = F)



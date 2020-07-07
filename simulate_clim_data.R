#==================#
# simulate climate #
#==================#
require(mgcv)

# 1. Create a TAir simulator (Northern & Southern Hemisphere)
## Sinusoidal function of average high and low temps + daily error model

TAir_simulator <- function(days = 730, Tmin, Tmax){
  amp = (Tmax-Tmin)/2
  avg = Tmin+amp
  if(Tmax > Tmin){
    # Northern Hemisphere
    temp <- sapply(1:days, function(x) amp * cos(2*pi/365*x + pi) + avg) + rnorm(days, mean = 0, sd=3)
  } else {
    # Sothern Hemisphere
    temp <- sapply(1:days, function(x) amp * cos(2*pi/365*x) + avg) + rnorm(days, mean = 0, sd=3)
  }
  return(temp)
}

minmax_temps <- function(Min = min(X$TAir), Max = max(X$TAir)){
  temps <- runif(2, min = Min, max = Max)
  return(temps)
}

get_temps <- function(days = 730, nsamples){
  temp = matrix(NA, nrow=nsamples, ncol=days)
  for (i in 1:nsamples){
    temps = minmax_temps()
    temp[i,] = TAir_simulator(Tmin = min(temps), Tmax = max(temps))
  }
  return(temp)
}


# 2. Multivariate regression: Model PAR, Precip, fAPAR and VPD by TAir and time.

## Fit model
#X_s = as.data.frame(scale(X[,1:6]))
#Y = cbind(X_s$PAR, X_s$VPD, X_s$Precip, X_s$fAPAR)

fit_mvr <- function(clim_data){

  fgam <- gam(list(PAR ~ s(TAir, bs="tp"),
          VPD ~ s(TAir, bs="tp") ,
          Precip ~ s(TAir, bs="tp"),
          fAPAR ~ s(TAir, bs="tp")), data = clim_data, family = mvn(d=4))

  #preds_gam <- predict(fgam, newdata = data.frame("TAir" = temp), type="response" )
  
  return(fgam)
}

simulate_climate <- function(nsamples, clim, days = 730, sim_CO2 = FALSE){
  
  if(sim_CO2 == FALSE){
    CO2 = X$CO2
  }
  
  temps = get_temps(nsamples = nsamples)
  fgam = fit_mvr(clim)
  
  arr = array(NA, dim = c(nsamples, days, 6))
  
  for(i in 1:nsamples){
    arr[i,,] <- cbind(temps[i,], predict(fgam, newdata=data.frame("TAir" = temps[i,]), type="response"), CO2)
  }
  
  return(arr)
}

#par(mfrow=c(2,3))
#for(j in 1:6){
#  plot(climsim[1,,j], type="l")
#    for (i in 2:nsamples) {
#    lines(climsim[i,,j], col="gray")  
#    }
#}

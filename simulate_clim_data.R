#==================#
# simulate climate #
#==================#

# 1. Create a TAir simulator (Northern Hemisphere)
## Sinusoidal function of average high and low temps + daily error model

TAir_simulator <- function(days = 730, Tmin, Tmax){
  amp = (Tmax-Tmin)/2
  avg = Tmin+amp
  temp <-  sapply(1:days, function(x) amp * cos(2*pi/365*x + pi) + avg) + rnorm(days, mean = 0, sd=3)
  return(temp)
}

temp = TAir_simulator(Tmin = min(X$TAir), Tmax = max(X$TAir))

plot(temp, col="red", type="l")
lines(X$TAir[1:365])


# 2. Multivariate regression: Model PAR, Precip, fAPAR and VPD by TAir and time.
plot(X[,1:6])
## Fit model
#X_s = as.data.frame(scale(X[,1:6]))
#Y = cbind(X_s$PAR, X_s$VPD, X_s$Precip, X_s$fAPAR)

Y = cbind(X$PAR, X$VPD, X$Precip, X$fAPAR)
fm = lm(Y ~ poly(TAir, 4) , data = X)
summary(fm)

preds = predict(fm, newdata = data.frame("TAir" = temp), type="response")

par(mfrow=c(2,2))
for(i in 1:4){
  plot(Y[,i], main = paste0("Y", i))
  points(preds[,i], col="red")
}

require(mgcv)
X$year = factor(rep(c(1,2), each=730/2))
X_TAir2 = X$TAir^2

fgam <- gam(list(Y[,1] ~ s(TAir, bs="tp"),
         Y[,2] ~ s(TAir, bs="tp") ,
         Y[,3] ~ s(TAir, bs="tp"),
         Y[,4] ~ s(TAir, bs="tp")), data = X, family = mvn(d=4))
summary(fgam)

preds_gam <- predict(fgam, newdata = data.frame("TAir" = temp, "DOY" = X$DOY), type="response" )
for(i in 1:4){
  plot(preds_gam[,i], col="red")
  points(Y[,i], main = paste0("Y", i))
  
}

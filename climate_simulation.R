require(mgcv)
require(lubridate)
require(mvtnorm)
require(ggplot2)

load("Rdata/profound/profound_in.RData")
X = X[X$site=="hyytiala",]

plot(X$DOY, type="l")

fmT = gam(TAir ~ s(DOY, by=year, bs = "cc"), data=X)
fmPAR = gam(PAR ~ s(DOY, by=year, bs = "cc"), data=X)
fmVPD = gam(VPD ~ s(DOY, by=year, bs = "cc"), data=X)
fmPrecip = gam(Precip ~ s(DOY, by=year, bs = "cc"), data=X)
fmfAPAR = gam(fAPAR ~ s(DOY, by=year, bs = "cc"), data=X)

save(fmT, file="Rdata/fmT.Rdata")
save(fmPAR, file="Rdata/fmPAR.Rdata")
save(fmVPD, file="Rdata/fmVPD.Rdata")
save(fmPrecip, file="Rdata/fmPrecip.Rdata")
save(fmfAPAR, file="Rdata/fmfAPAR.Rdata")

summary(fmT)
summary(fmPAR)
summary(fmVPD)
summary(fmPrecip)
summary(fmfAPAR)

plot(fmT$residuals)
plot(fmPAR$residuals)
plot(fmVPD$residuals)
plot(fmPrecip$residuals)
plot(fmfAPAR$residuals)

T_hat = predict(fmT, data.frame(DOY = 1:365, year=2001))
PAR_hat = predict(fmPAR, data.frame(DOY = 1:365, year=2001))
VPD_hat = predict(fmVPD, data.frame(DOY = 1:365, year=2001))
Precip_hat = predict(fmPrecip, data.frame(DOY = 1:365, year=2001))
fAPAR_hat = predict(fmfAPAR, data.frame(DOY = 1:365, year=2001))

plot(1:365, T_hat, type="l")
plot(1:365, PAR_hat, type="l")
plot(1:365, VPD_hat, type="l")
plot(1:365, Precip_hat, type="l")
plot(1:365, fAPAR_hat, type="l")

res_mat = data.frame(TAir=fmT$residuals, 
                     PAR=fmPAR$residuals, 
                     VPD=fmVPD$residuals, 
                     Precip=fmPrecip$residuals, 
                     fAPAR = fmfAPAR$residuals)

plot(res_mat)
summary(res_mat)

cov_mat = cov(res_mat)

noise = rmvnorm(365, mean=rep(0, length=ncol(res_mat)), sigma=cov_mat)
summary(noise)
plot(1:365, noise[,1])

T_hat = T_hat + noise[,1]
PAR_hat = PAR_hat + noise[,2]
VPD_hat = VPD_hat + noise[,3]
Precip_hat = Precip_hat + noise[,4]
fAPAR_hat = fAPAR_hat + noise[,5]

PAR_hat[which(PAR_hat<0)] = 0
VPD_hat[which(VPD_hat<0)] = 0
Precip_hat[which(Precip_hat<0)] = 0
fAPAR_hat[which(fAPAR_hat<0)] = 0

vars = factor( rep(c("TAir", "PAR", "VPD", "Precip", "fAPAR"), each = 365),  levels= c("TAir", "PAR", "VPD", "Precip", "fAPAR"))
plot_data = data.frame(climsim = c(T_hat, PAR_hat, VPD_hat, Precip_hat, fAPAR_hat), 
                       var = vars,
           climobs = c(X$TAir[X$year==2001], X$PAR[X$year==2001], X$VPD[X$year==2001], X$Precip[X$year==2001], X$fAPAR[X$year==2001]),
           DOY = rep(1:365, times=5))

p = ggplot(plot_data) +
    geom_line(aes(x=DOY, y=climobs, color="Observation")) +
    geom_line(aes(x=DOY, y=climsim, color="Simulation"), alpha=0.8) +
    facet_wrap(.~var, scales = "free") + 
    scale_color_manual(name="",values= c("darkblue", "orange")) +
    ylab("Value in respective Unit") +
    theme_light(base_size = 13) +
    theme(legend.position = "top")

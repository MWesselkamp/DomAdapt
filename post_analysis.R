#======================#
# Effect Plots: PRELES #
#======================#

library(Rpreles)


profound_in <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/profound/profound_in", sep=";")
load("~/Sc_Master/Masterthesis/Project/DomAdapt/Rdata/CalibratedParametersHytProf.Rdata")

df = profound_in[profound_in$site=="hyytiala",]

new_df = data.frame(TAir=seq(min(df$TAir), max(df$TAir), length.out = 544), PAR = mean(df$PAR), VPD = mean(df$VPD), Precip=mean(df$Precip), fAPAR = mean(df$fAPAR), DOY=mean(df$DOY))

GPP_preds = 1:nrow(new_df)
for (i in 1:nrow(new_df)){
  GPP_preds[i] <- unlist(PRELES(PAR=new_df$PAR[i],TAir=new_df$TAir[i],VPD=new_df$VPD[i],Precip=new_df$Precip[i],CO2=380,fAPAR=new_df$fAPAR[i],p=par$calib[1:30],returncols = c("GPP")))
}

plot(GPP_preds, type = "l")

#======================#
# Climate simulations  #
#======================#

library(extrafont)
font_import()
loadfonts(device = "win")

source("simulation_functions.R")

load("Rdata/fmT.Rdata")
load("Rdata/fmPAR.Rdata")
load("Rdata/fmVPD.Rdata")
load("Rdata/fmPrecip.Rdata")
load("Rdata/fmfAPAR.Rdata")

summary(fmT)

T_hat = predict(fmT, data.frame(DOY = 1:366, year=2008))
PAR_hat = predict(fmPAR, data.frame(DOY = 1:366, year=2008))
VPD_hat = predict(fmVPD, data.frame(DOY = 1:366, year=2008))
Precip_hat = predict(fmPrecip, data.frame(DOY = 1:366, year=2008))
fAPAR_hat = predict(fmfAPAR, data.frame(DOY = 1:366, year=2008))

climsims = NULL
for (i in 1:20){
  climsims = rbind(climsims, climate_simulator(365, 1))
}

vars = c("TAir", "PAR", "VPD", "Precip", "fAPAR")

cm = climsims %>% 
    group_by(DOY) %>% 
    summarize_at("TAir", list(~mean(.), ~sd(.))) 
names(cm) = c("DOY", "m", "s")


ggplot(cm, aes(x=DOY, y=m)) +
  geom_ribbon(aes(ymin = m-2*s, ymax=m+2*s), fill="gray")+
  geom_line(aes(x=DOY, y = X$TAir[X$year==2008], color="Observed"), color="red", lwd=1.2) +
  ylab("Mean temperature [degree Celsius]") +
  xlab("Day of year") +
  theme_bw()+
  theme(panel.background = element_blank(), 
        panel.grid = element_blank(), 
        axis.text = element_text(size=20, family="Palatino Linotype", color = "black"), 
        axis.title = element_text(size=20, family="Palatino Linotype", color = "black"),
        aspect.ratio=1) 


cm = climsims %>% 
  group_by(DOY) %>% 
  summarize_at("PAR", list(~mean(.), ~sd(.))) 
names(cm) = c("DOY", "m", "s")

ggplot(cm, aes(x=DOY, y=m)) +
  geom_ribbon(aes(ymin = m-2*s, ymax=m+2*s), fill="gray")+
  geom_line(aes(x=DOY, y = X$PAR[X$year==2008], color="Observed"), color="red", lwd=1.2) +
  ylab("Mean temperature [degree Celsius]") +
  xlab("Day of year") +
  theme_bw()+
  theme(panel.background = element_blank(), 
        panel.grid = element_blank(), 
        axis.text = element_text(size=20, family="Palatino Linotype", color = "black"), 
        axis.title = element_text(size=20, family="Palatino Linotype", color = "black"),
        aspect.ratio=1) 


cm = climsims %>% 
  group_by(DOY) %>% 
  summarize_at("VPD", list(~mean(.), ~sd(.))) 
names(cm) = c("DOY", "m", "s")

ggplot(cm, aes(x=DOY, y=m)) +
  geom_ribbon(aes(ymin = m-2*s, ymax=m+2*s), fill="gray")+
  geom_line(aes(x=DOY, y = X$VPD[X$year==2008], color="Observed"), color="red", lwd=1.2) +
  ylab("Mean temperature [degree Celsius]") +
  xlab("Day of year") +
  theme_bw()+
  theme(panel.background = element_blank(), 
        panel.grid = element_blank(), 
        axis.text = element_text(size=20, family="Palatino Linotype", color = "black"), 
        axis.title = element_text(size=20, family="Palatino Linotype", color = "black"),
        aspect.ratio=1) 


#=================================#
# Plot simulations: Climate Fixed #
#=================================#

sims_in <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/simulations/climateFix/sims_in.csv", header=TRUE, sep=";")
sims_out <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/simulations/climateFix/sims_out.csv", header=TRUE, sep=";")

summary(sims_in)
summary(sims_out)

sims_in$sample = rep(1:365, each = 1000)
sims_out$DOY = sims_in$DOY
sims_out$sample = sims_in$sample

sims_ci = sims_out %>% 
  group_by(DOY) %>% 
  summarise(med = mean(GPP), ci_low = quantile(GPP, c(0.05)), ci_up = quantile(GPP, c(0.95)))

require(ggplot2)

ggplot(sims_in)+
  geom_ribbon(data = sims_ci, aes(ymin = ci_low, ymax = ci_up, x= DOY), fill="salmon", alpha = 0.8) +
  geom_path(data=sims_ci, aes(x=DOY, y=med)) +
  ylab("Predicted GPP") +
  theme_light()


sims_in_gg = gather(sims_in, "climvar", "Value", 1:5)
sims_in_gg = sims_in_gg[which(sims_in_gg$sample==1),]
sims_in_gg$climvar = factor(sims_in_gg$climvar, levels = c("TAir", "PAR", "VPD", "Precip", "fAPAR"))

ggplot(sims_in_gg) +
  geom_line(aes(x = DOY, y=Value, group = sample)) +
  facet_wrap(.~climvar, scales = "free") +
  xlab("Day of Year") +
  theme_light()

sims_in_pars = gather(sims_in, "par", "Value", 10:14)
sims_in_pars = sims_in %>% 
  select(c("beta", "X0", "gamma", "alpha", "chi", "sample"))  %>% 
  group_by(sample) %>% 
  summarise_all(list(~ mean(.)))
sims_in_pars = gather(sims_in_pars, "par", "Value", 2:6)

ggplot(sims_in_pars) +
  geom_histogram(aes(x= Value), stat="bin", bins=40, fill="lightblue", color="black", alpha=0.7) +
  facet_wrap(.~par, scales = "free") +
  theme_light()

#=================================#
# Plot simulations: Params  Fixed #
#=================================#

sims_in <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/simulations/paramsFix/sims_in.csv", header=TRUE, sep=";")
sims_out <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/simulations/paramsFix/sims_out.csv", header=TRUE, sep=";")

summary(sims_in)

sims_in$sample = rep(1:365, each = 1000)
sims_out$DOY = sims_in$DOY
sims_out$sample = sims_in$sample

sims_ci = sims_out %>% 
  group_by(DOY) %>% 
  summarise(med = mean(GPP), ci_low = quantile(GPP, c(0.05)), ci_up = quantile(GPP, c(0.95)))

ggplot(sims_in)+
  geom_ribbon(data = sims_ci, aes(ymin = ci_low, ymax = ci_up, x= DOY), fill="salmon", alpha = 0.8) +
  geom_path(data=sims_ci, aes(x=DOY, y=med)) +
  ylab("Predicted GPP") +
  theme_light()

sims_clim = sims_in %>% 
  group_by(DOY) %>% 
  summarise_at(c("TAir", "PAR", "VPD", "Precip", "fAPAR"), 
               list(mu = ~mean(.), q1= ~quantile(., 0.05),q2=~quantile(., 0.95)))

sims_clim_gg = gather(sims_clim, "Var_mean", "mean", 2:6)
sims_clim_gg = gather(sims_clim_gg, "Var_lower", "lower", 2:6)
sims_clim_gg = gather(sims_clim_gg, "Var_upper", "upper", 2:6)
sims_clim_gg = gather(sims_clim_gg, "Vars", "Names", c(2,4,6))
sims_clim_gg$Vals = sapply(strsplit(sims_clim_gg$Vals, "_"), "[", 1)
sims_clim_gg = gather(sims_clim_gg, "quant", "vals", c(2,3,4))

ggplot(sims_clim_gg) +
  geom_line(aes(x= DOY, y=vals, group = quant, colour=quant), colour="salmon", alpha = 0.8) +
 #geom_line(aes(x = DOY, y=mean, group = Vars)) +
  facet_wrap(.~Vals, scales = "free")
  #xlab("Day of Year") +
  #theme_light()

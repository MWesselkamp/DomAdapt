#=================#
# run simulations #
#=================#

# In this file, the climate data to use and the number of simulations are specified.
# As well as the range of functions we are going to use below.
source("simulation_functions.R")

load("Rdata/fmT.Rdata")
load("Rdata/fmPAR.Rdata")
load("Rdata/fmVPD.Rdata")
load("Rdata/fmPrecip.Rdata")
load("Rdata/fmfAPAR.Rdata")

nsamples = 1000
seq_len=365

data_dir <- "data/preles/simulations/"

pars = get_parameters()
pars_values = pars$Default
pars_names = c("beta", "X0", "gamma", "alpha", "chi")

sims_in = NULL
sims_out = NULL

for (sample in 1:nsamples){
  
  climate_simulations = climate_simulator(seq_len, sample)
  
  pars_lhs = sample_parameters(pars = pars)
  
  pars_values[which(as.character(pars$Name) %in% pars_names)] = pars_lhs[,1]
  
  targets = matrix(unlist(get_preles_output(climate_simulations, pars_values, c("GPP"))), nrow = seq_len, ncol=1)
  
  features = cbind(climate_simulations, apply(pars_lhs, 1, function(x) rep(x, times=seq_len)))
  
  names(features)[10:14] = pars_names
  
  sims_in = rbind(sims_in, features)
  sims_out = rbind(sims_out, targets)
  
}

write.table(sims_in, file=paste0(data_dir, "sims_in.csv"), sep=";", row.names = FALSE)
write.table(sims_out, file=paste0(data_dir, "sims_out.csv"), sep=";", row.names = FALSE, col.names = c("GPP"))

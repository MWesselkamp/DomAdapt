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

params = get_parameters()

run_sim = function(nsamples, days, params_distr, fix, pars = params, data_dir = "data/simulations/"){
  
  pars_values = pars$Default
  pars_names = c("beta", "X0", "gamma", "alpha", "chi")
  
  sims_in = NULL
  sims_out = NULL
  
  pars_lhs = sample_parameters(pars, nsamples, params_distr)

  climate_simulations = climate_simulator(days, 1)

  for (sample in 1:nsamples){
    
    if (fix!="climate"){ climate_simulations = climate_simulator(days, sample) }
  
    if (fix !="params"){pars_values[which(as.character(pars$Name) %in% pars_names)] = pars_lhs[,sample]}
  
    targets = matrix(unlist(get_preles_output(climate_simulations, pars_values, c("GPP"))), nrow = days, ncol=1)
  
    features = cbind(climate_simulations, apply(as.matrix(pars_values[which(as.character(pars$Name) %in% pars_names)]), 1, function(x) rep(x, times=days)))
  
    names(features)[10:14] = pars_names
  
    sims_in = rbind(sims_in, features)
    sims_out = rbind(sims_out, targets)
  
  }
  
  if (fix == "None"){
    write.table(sims_in, file=paste0(data_dir, params_distr, "_params/sims_in.csv"), sep=";", row.names = FALSE)
    write.table(sims_out, file=paste0(data_dir, params_distr, "_params/sims_out.csv"), sep=";", row.names = FALSE, col.names = c("GPP"))
  }else{
    write.table(sims_in, file=paste0(data_dir, fix, "Fix/sims_in.csv"), sep=";", row.names = FALSE)
    write.table(sims_out, file=paste0(data_dir, fix, "Fix/sims_out.csv"), sep=";", row.names = FALSE, col.names = c("GPP"))
  }
}

run_sim(1000, 365, "normal", "params")

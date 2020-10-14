#=================#
# run simulations #
#=================#

# In this file, the climate data to use and the number of simulations are specified.
# As well as the range of functions we are going to use below.
source("simulation_functions.R")

nsamples = 100
# get the values for the 30 Preles parameters:
#   default values or calibrated parameters?
pars_def <- get_parameters(default = TRUE)
pars_calib <- get_parameters(default= FALSE)

# sample a selection of these parameters (here: 5) in a latin hypercube design
parsLHS <- sample_parameters(pars_default = pars_calib, nsamples = nsamples, pars_names = c("beta", "X0", "gamma", "alpha", "chi"))

# Simulate climate

climate_simulations = climate_simulator(nsamples = nsamples, seq_len = 365)

#fgam = fit_mvr(X)
#save(fgam, file="Rdata/fgam_mvr.Rdata")

load("Rdata/fgam_mvr.Rdata")

climate_simulation <- simulate_climate(nsamples = nsamples, fgam = fgam, days = 365)
save(climate_simulation, file="Rdata/climate_simulation.Rdata")

# create Preles output from climate data and the parameter combinations.
# In the same step, write the output to files (in data/profound) for use in Python. 
# Save the output for visualization in R.
data_dir <- "data/preles/simulations/"
get_simulations(pars = pars_calib, clim_data = climate_simulation, data_dir = data_dir)

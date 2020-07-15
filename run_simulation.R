#=================#
# run simulations #
#=================#

# In this file, the climate data to use and the number of simulations are specified.
# As well as the range of functions we are going to use below.
source("preles_simulations.R")
source("simulate_clim_data.R")

nsamples = 100
# get the values for the 30 Preles parameters:
#   default values or calibrated parameters?
pars_def <- get_parameters(default = TRUE)
pars_calib <- get_parameters(default= FALSE)

# sample a selection of these parameters (here: 5) in a latin hypercube design
parsLHS <- sample_parameters(pars_default = pars_calib, nsamples = nsamples, pars_names = c("beta", "X0", "gamma", "alpha", "chi"))

# Load data 
#   Climatic input for four boreal sites in Finland descending from the Master Thesis of Elias Schneider (based on Minunno et al. (2016)):
load("Rdata/borealsites/EddyCovarianceDataBorealSites.RData")
#   Climatic input extracted from the Profound data base.
load("Rdata/profound/profound_in.RData")

# choose the data set to use (X or s1-4)
X <- X[,-7] # profound input
X$CO2 <- 380.8275

climate_data <- simulate_climate(nsamples = nsamples, clim = X)

# create Preles output from climate data and the parameter combinations.
# In the same step, write the output to files (in data/profound) for use in Python. 
# Save the output for visualization in R.
data_dir <- "data/preles/pars_calibrated/"
get_simulations(pars = pars_calib, clim_data = climate_data, data_dir = data_dir)

# Plot model predictions and s1 observations.
pdf("plots/preles_output.pdf")
plot_preles_output(output = output, nsamples = samples, all = F)
dev.off()

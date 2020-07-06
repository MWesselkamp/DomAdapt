#=================#
# run simulations #
#=================#

# In this file, the climate data to use and the number of simulations are specified.
# As well as the range of functions we are going to use below.
source("preles_simulations.R")

# get the values for the 30 Preles parameters:
#   default values or calibrated parameters?
pars_def <- get_parameters(default = TRUE)
pars_calib <- get_parameters(default= FALSE)

# sample a selection of these parameters (here: 5) in a latin hypercube design
parsLHS <- sample_parameters(pars_default = pars_calib, nsamples= 10000, pars_names = c("beta", "X0", "gamma", "alpha", "chi"))

# Load data 
#   Climatic input for four boreal sites in Finland descending from the Master Thesis of Elias Schneider (based on Minunno et al. (2016)):
load("Rdata/borealsites/EddyCovarianceDataBorealSites.RData")
#   Climatic input extracted from the Profound data base.
load("Rdata/profound/profound_input.RData")

# choose the data set to use (X or s1-4)
climate_data <- X[,-7] # profound input
climate_data$CO2 <- 380.8275

# create Preles output from climate data and the parameter combinations.
# In the same step, write the output to files (in data/profound) for use in Python. 
# Save the output for visualization in R.
data_dir <- "data/preles/pars_calibrated/"
output <- get_lhs_output(pars = pars_calib, data_dir = data_dir)
save(output, file="data/preles/pars_calibrated/output.Rdata")

# write the input data (cliamte+parsLHS) to files for use in Pyhton.
write_input_data(data_dir = data_dir)

# Plot model predictions and s1 observations.
pdf("plots/preles_output.pdf")
plot_preles_output(output = output, nsamples = samples, all = F)
dev.off()

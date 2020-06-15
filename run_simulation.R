#=================#
# run simulations #
#=================#

# In this file, the climate data to use and the number of simulations are specified.
# As well as the range of functions we are going to use below.
source("preles_simulations.R")

# get the default value for the 30 Preles parameters.
pars_def <- get_default_parameters()

# sample a selection of these parameters (here: 5) in a latin hypercube design
parsLHS <- sample_parameters(pars_default = pars_def, pars_names = c("beta", "X0", "gamma", "alpha", "chi"))

# create Preles output from climate data and the parameter combinations.
# In the same step, write the output to files (in data/profound) for use in Python. 
# Save the output for visualization in R.
output <- get_lhs_output()
save(output, file="Rdata/profound/output.Rdata")

# write the input data (cliamte+parsLHS) to files for use in Pyhton.
write_input_data()

# Plot model predictions and s1 observations.
pdf("plots/preles_output.pdf")
plot_preles_output(output = output, nsamples = samples, all = F)
dev.off()

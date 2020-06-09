# =================#
# Helper functions #
# =================#

# a function that passes the input data (climate and parameters) to the RPreles function.
get_output <- function(clim, params){
  
  # the function takes:
  #   Climate data containing the variables PAR, TAir, VPS, Precip, CO2, fAPAR
  # it eturns:
  #   Preles output, containing variables GPP, ET, SW.
  
  output <- PRELES(PAR = clim$PAR, TAir = clim$TAir, VPD = clim$VPD, Precip = clim$Precip, CO2 = clim$CO2, fAPAR = clim$fAPAR, p = params)
  
  return(output)
  
}

# function that plots the output data returned by Rpreles.
plot_output <- function(output, vars = c("GPP", "EV", "SW"), all=FALSE){
  
  # the function takes:
  #   list of length nsamples, containing lists of output variables of length nrow(s1).
  #   character vector containing variable names
  #   logical argument, if all variables will be plotted. Default to FALSE (only GPP plot)
  # the function returns:
  #   a lines plot of output data. Default is GPP, but can be changed such that all are plotted as facets.
  
  output <- as.data.frame(matrix(unlist(output), nrow=nsamples*nrow(s1), ncol = 3))
  names(output) <- vars
  
  output <- output %>% 
    mutate(sim = rep(1:nsamples, each = nrow(s1)),
           DOY = rep(1:nrow(s1), times = nsamples))
  
  if(all){
    output <- output %>% 
      gather(key = "Var", value = "Value", 1:length(vars))
    
    ggplot(output) +
      geom_path(aes(x = DOY, y = Value, group = sim)) +
      facet_wrap(.~Var)
  } else {
  
    ggplot(output) +
      geom_path(aes(x = DOY, y = GPP, group = sim))
  }
}

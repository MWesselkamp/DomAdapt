# =================#
# Helper functions #
# =================#

# a function that passes the input data (climate and parameters) to the RPreles function.
get_preles_output <- function(clim, params, return_cols){
  
  # the function takes:
  #   Climate data containing the variables PAR, TAir, VPS, Precip, CO2, fAPAR
  # it eturns:
  #   Preles output, containing variables GPP, ET, SW.
  
  output <- PRELES(PAR = clim$PAR, TAir = clim$TAir, VPD = clim$VPD, Precip = clim$Precip, CO2 = clim$CO2, fAPAR = clim$fAPAR, p = params, returncols = return_cols)
  
  return(output)
  
}

# function that plots the output data returned by Rpreles.
plot_preles_output <- function(output, nsamples, vars = c("GPP", "SW"), all=FALSE){
  
  # the function takes:
  #   list of length nsamples, containing lists of output variables of length nrow(s1).
  #   character vector containing variable names
  #   logical argument, if all variables will be plotted. Default to FALSE (only GPP plot)
  # the function returns:
  #   a lines plot of output data. Default is GPP, but can be changed such that all are plotted as facets.
  if(nsamples==1){
    len_output <- length(output[[1]])
  }else{
    len_output <- length(output[[1]][[1]])
  }
  
  output <- as.data.frame(matrix(unlist(output), nrow=length(unlist(output)), ncol = 1))
  names(output) <- "Value"
  
  output <- output %>% 
    mutate(var=rep(rep(vars, each=len_output), times=nsamples),
          sim = rep(1:nsamples, each = len_output*2),
          DOY = rep(1:len_output, times = nsamples*2))
  
  if(all){
    
    p <- ggplot(output) +
      geom_path(aes(x = DOY, y = Value, group = sim))+
      facet_wrap(.~var)
    
  } else {
    p <- ggplot(output[output$var=="GPP",]) +
      geom_path(aes(x = DOY, y = Value, group = sim))+
      facet_wrap(.~var)

  }
  p
}

#===============#
# Plot data #
#===============#
library(ggplot2)

# Plot the climatic measurements or the state variables of stands.
plot_climate = function(vars, data){
  
  n_sites = length(unique(X$site))
  data$Time = rep(1:(nrow(data)/n_sites), times = n_sites)
  
  for (i in 1:length(vars)){
    
    var = vars[i]
    
    p = ggplot(data) + 
      geom_line(aes(x= Time, y=data[,c(var)], group=site, colour="Observations")) +
      facet_wrap(.~site, scales = "free")  +
      scale_color_manual(name="", values=c("blue")) +
      ylab(var) +
      theme_bw() +
      theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), legend.position = "none")
    
    png(paste0("plots/data_",var, ".png"), width = 900, height=700)
    print(p)
    dev.off()
  }
  
}



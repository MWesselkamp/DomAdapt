#plot_simulation
require(ggplot2)

sims_lon <- gather(sims_in, "variable", "value", c("TAir", "PAR", "VPD", "Precip", "fAPAR"))


ggplot(sims_lon) +
  geom_line(aes(DOY, value, color=sample, group=sample), alpha=0.6) +
  facet_wrap(.~variable, scales = "free") +
  theme_light()

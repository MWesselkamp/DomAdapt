# load eddy covariance data
load("Rdata/borealsites/EddyCovarianceDataBorealSites.RData")
source("visualization.R")

s2 <- s2[,1:9]
s3 <- s3[,1:9]
s4 <- s4[,1:9]
n = c("PAR","TAir","VPD","Precip","CO2","fAPAR","GPP", "ET","DOY") 
names(s1) <- n
names(s2) <- n
names(s3) <- n
names(s4) <- n

s1$site = "hyytiala"
s2$site = "sodankyl"
s3$site = "alkkia"
s4$site = "kalevansuo"

boreal_sites_in = do.call(rbind, list(s1, s2, s3, s4))
boreal_sites_out = as.data.frame(boreal_sites_in[,c("GPP", "ET")])
boreal_sites_in = boreal_sites_in[,c(1:6, 9:10)]

# Remove missing values:
boreal_sites_in[boreal_sites_in <= -990.0] = NA

# Plot to see what plots to use for training and validation sites
plot_climate(c("GPP", "ET"), cbind(boreal_sites_in,boreal_sites_out))
vars = c("PAR", "TAir", "VPD", "Precip", "fAPAR")
plot_climate(vars, data = boreal_sites_in)
# Remove Alkkia
alkkia = which(boreal_sites_in$site=="alkkia")
boreal_sites_in = boreal_sites_in[-alkkia,]
boreal_sites_out = boreal_sites_out[-alkkia,]

save(boreal_sites_in, file="Rdata/borealsites/boreal_sites_in.Rdata")
write.table(boreal_sites_in, file="data/borealsites/boreal_sites_in", sep = ";", row.names = FALSE)
save(boreal_sites_out, file="Rdata/borealsites/boreal_sites_out.Rdata")
write.table(boreal_sites_out, file="data/borealsites/boreal_sites_out", sep = ";", row.names = FALSE)

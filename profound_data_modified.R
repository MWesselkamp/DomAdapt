library(lubridate)

load("Rdata/profound/profound_out.Rdata")
load("Rdata/profound/profound_in.Rdata")


#==================================================#
# Save test data set ("le bray") to different file #
#==================================================#

lebray = which(X$site == "le_bray")
X_test = X[lebray,]
y_test = data.frame(GPP = y[lebray,])
X = X[-lebray,]
y = data.frame(GPP = y[-lebray,])

save(y_test, file="Rdata/profound/profound_out_test.Rdata")
write.table(y_test, file="data/profound/profound_out_test", sep = ";",row.names = FALSE)
save(X_test, file="Rdata/profound/profound_in_test.Rdata")
write.table(X_test, file="data/profound/profound_in_test", sep = ";", row.names = FALSE)

#===========================#
# Detect missing GPP values #
#===========================#


# filter years where GPP measurements are not available
# (hyytiala 2007 and le_bray 2002)
GPPavg = y %>% 
  group_by(site = X$site, year = year(date(X$date))) %>% 
  summarise(avg = mean(GPP)) %>% 
  filter(avg == 0)

# remove selection from X and y.
rem = which(((year(date(X$date)) %in% GPPavg$year) & (X$site %in% GPPavg$site)))
X = X[-rem,]
y = data.frame(GPP= y[-rem,])

save(y, file="Rdata/profound/profound_out_trainval.Rdata")
write.table(y, file="data/profound/profound_out_trainval", sep = ";",row.names = FALSE)
save(X, file="Rdata/profound/profound_in_trainval.Rdata")
write.table(X, file="data/profound/profound_in_trainval", sep = ";", row.names = FALSE)
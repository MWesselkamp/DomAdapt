#==================#
# Calibrate Preles #
#==================#


library(Rpreles)
library(lhs)
library(coda)
library(BayesianTools)

# Load default parameter values.
load("~/Sc_Master/Masterthesis/Project/DomAdapt/Rdata/ParameterRangesPreles.RData")

parind <- c(5:10, 14:16) # Indexes for PRELES parameters

profound_in <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/profound/profound_in", sep=";")
profound_out <- read.csv("~/Sc_Master/Masterthesis/Project/DomAdapt/data/profound/profound_out", sep=";")

hyt = which(((profound_in$site == "hyytiala") & (profound_in$year %in% c(2001,2003,2004,2005,2006))))

profound_in = profound_in[hyt,]
profound_out = profound_out[hyt,]

#1- Likelihood function
likelihood <- function(pValues){
  p <- par$def
  p[parind] <- pValues # new parameter values
  predicted<- PRELES(DOY=profound_in$DOY,PAR=profound_in$PAR,TAir=profound_in$TAir,VPD=profound_in$VPD,Precip=profound_in$Precip,
                     CO2=profound_in$CO2,fAPAR=profound_in$fAPAR,p=p[1:30])
  diff_GPP <- predicted$GPP-profound_out
  llvalues <- sum(dnorm(predicted$GPP, mean = profound_out, sd = p[31], log=T)) ###   llvalues <- sum(dnorm(diff_GPP, sd = p[31], log=T))
  #llvalues <- sum(dexp(abs(diff_GPP),rate = 1/(p[31]+p[32]*predicted$GPP),log=T))
  return(llvalues)
}

#2- Prior
prior <- createUniformPrior(lower = par$min[parind], upper = par$max[parind])

#=Bayesian set up=#

BSpreles <- createBayesianSetup(likelihood, prior, best = par$def[parind], names = par$name[parind], parallel = F)

bssetup <- checkBayesianSetup(BSpreles)

#=Run the MCMC with three chains=#

settings <- data.frame(iterations = 1e5, optimize=F, nrChains = 3)
chainDE <- runMCMC(BSpreles, sampler="DEzs", settings = settings)
par.opt<-MAP(chainDE) #gets the optimized maximum value for the parameters

# Check convergence:
tracePlot(chainDE, parametersOnly = TRUE, start = 1, whichParameters = 1:4)
tracePlot(chainDE, parametersOnly = TRUE, start = 1, whichParameters = 5:9)

marginalPlot(chainDE, scale = T, best = T, start = 5000)
correlationPlot(chainDE, parametersOnly = TRUE, start = 2000)

# save calibrated parameters
par$calib = par$def
par$calib[parind] = par.opt$parametersMAP
save(par, file = "~/Sc_Master/Masterthesis/Project/DomAdapt/Rdata/CalibratedParametersHytProf.Rdata")

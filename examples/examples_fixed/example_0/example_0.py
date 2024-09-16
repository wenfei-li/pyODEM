""" Basic example of a run through using the methods in this package"""

##import the basic things
import numpy as np
import pyODEM.observables as observables
import math

#load a histogram data
edges = np.loadtxt("edges.dat")
obs = observables.ExperimentalObservables()
obs.add_histogram("exp_data.dat", edges=edges, errortype="gaussian", scale = 1000.0) #load and format the data distribution

# Case 1 : calculate Q score based on histogram counts

qcalc = obs.get_q_functions()[0] ##first function is Q-funciton. Second function is the derivative function.
logqcalc = obs.get_log_q_functions()[0]

regdata = np.loadtxt('hist.dat')

normalization = np.sum(regdata*np.diff(edges))
regdata_normalized = regdata / normalization
print(qcalc(regdata),logqcalc(regdata))

regdata[0] = 10
normalization = np.sum(regdata*np.diff(edges))
regdata_normalized = regdata / normalization
print(qcalc(regdata),logqcalc(regdata))

regdata[2] = 282.0
normalization = np.sum(regdata*np.diff(edges))
regdata_normalized = regdata / normalization
print(qcalc(regdata),logqcalc(regdata))

# Case 2 : obtain the histogram count based on data points, then calculate Q score
# I will generate a random sample following the distribution provided by exp_data.dat
nframe = 100

exp_data = np.loadtxt('exp_data.dat')
raw_data = np.random.choice(5,nframe,p=exp_data[:,0]).reshape(1,nframe)
all_obs,all_std = obs.compute_observations(raw_data)

print(qcalc(all_obs),logqcalc(all_obs))

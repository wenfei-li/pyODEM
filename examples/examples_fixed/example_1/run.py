""" Example for analyzing a 1d langevin dynamics model

This script analyzes the langevin 1-d run and outputs a new set of parameters
based on the simulation run.
"""
import numpy as np
import scipy.stats as stats
import os

import pyODEM
ml = pyODEM.model_loaders
observables = pyODEM.observables
ene = pyODEM.estimators.max_likelihood_estimate


#load the model and load the data per the model's load_data method.
lmodel = ml.Langevin("simple.ini")
iteration = lmodel.model.iteration
data = lmodel.load_data("iteration_%d/position.dat" % iteration)

#load the observable object that calculates the observables of a set of simulation data
edges = np.loadtxt("edges.dat")
obs = observables.ExperimentalObservables()
obs.add_histogram("exp_data.dat", edges=edges, errortype="gaussian", scale = 10) #load and format the data distribution

#do a simple discretizaiton fo the data into equilibrium distribution states.
#In theory, the user will be able to specify any sort of equlibrium states for their data
hist, tempedges, slices = stats.binned_statistic(data, np.ones(np.shape(data)[0]), bins = 200, statistic="sum")
possible_slices = np.arange(np.min(slices), np.max(slices)+1)

equilibrium_frames = {}
indices = np.arange(np.shape(data)[0])
for i in possible_slices:
    state_data = indices[slices == i]
    if not state_data.size == 0:
        equilibrium_frames[i] = state_data

formatted_data = []
for i in possible_slices:
    data_single = {}
    data_single["index"] = i
    data_single["data"] = data[equilibrium_frames[i]]
    obs_result, obs_std = obs.compute_observations(data_single["data"].reshape(1,len(equilibrium_frames[i])))
    data_single["obs_result"] = obs_result
    formatted_data.append(data_single)

#Now we can compute the set of epsilons that satisfy the max-likelihood condition
#set logq=True for using the logarithm functions
#currently, options are simplex, annealing, cg
solutions = ene(formatted_data, observables=obs, solver="simplex", model=lmodel, logq=False)
new_eps = solutions.new_epsilons
old_eps = solutions.old_epsilons
Qold = solutions.oldQ
Qnew = solutions.newQ
Qfunction = solutions.Qfunction_epsilon #-Q function
Qfunction_log = solutions.log_Qfunction_epsilon #-log(Q) function

print("Epsilons are: ")
print(new_eps)
print(old_eps)

print("")
print("Qold: %g" %Qold)
print("Qnew: %g" %Qnew)

savestr = "iteration_%d/newton" % iteration
if not os.path.isdir(savestr):
    os.mkdir(savestr)
np.savetxt("%s/params" % savestr, np.append([1.0], new_eps))

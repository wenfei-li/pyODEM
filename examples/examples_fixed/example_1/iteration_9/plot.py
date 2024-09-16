import numpy
import matplotlib.pyplot as plt

edge_ref = numpy.loadtxt('../edges.dat')
data_ref = numpy.loadtxt('../exp_data.dat')[:,0]
plt.stairs(data_ref,edge_ref,linewidth=3,color='green',zorder=10)

data = numpy.loadtxt('position.dat')
plt.hist(data, bins=100, density=True, color='skyblue', alpha=0.5, edgecolor='skyblue',zorder=5)

plt.savefig('hist.png')

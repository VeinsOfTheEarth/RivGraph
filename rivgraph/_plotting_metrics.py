# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:35:48 2018

@author: Jon
"""

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import os, pickle


result_names = ['Colville', 'Kolyma', 'Mackenzie', 'Yenisei', 'Yukon']
result_paths = [os.path.join(r"X:\RivGraph\Results", rn, rn + '_metrics.pkl') for rn in result_names]



""" Nonlinear entropy rates """

# Initialize figure
fig, axs = plt.subplots(len(result_names), 1, sharex=True, sharey=True)

for i, (d, dp) in enumerate(zip(result_names, result_paths)):
    print(dp)
    with open(dp, 'rb') as f:
        deltavars = pickle.load(f)

    N = len(deltavars['nER_randomized'])
    axs[i].hist(np.array(deltavars['nER_randomized'])/N)
    axs[i].plot(deltavars['nonlin_entropy_rate']/N, 0, '*', markersize=10)
    axs[i].set_title(d + ', pExc={}'.format(deltavars['nER_prob_exceedence']))


""" Nap vs LI """
plt.close('all')
fig, ax = plt.subplots()
for i, (d, dp) in enumerate(zip(result_names, result_paths)):
    
    with open(dp, 'rb') as f:
        deltavars = pickle.load(f)

    LI = deltavars['leakage_idx'][:,1]
    li25 = np.percentile(LI, 25)
    li50 = np.percentile(LI, 50)
    li75 = np.percentile(LI, 75)
    
    nap = deltavars['n_alt_paths'][:,1]
    nap25 = np.percentile(nap, 25)
    nap50 = np.percentile(nap, 50)
    nap75 = np.percentile(nap, 75)

    lines = [[(li25,nap50), (li75,nap50)], [(li50, nap25),(li50, nap75)]]
    c = np.random.rand(3,)

    lc = mc.LineCollection(lines, colors=c, linewidths=2)
    ax.add_collection(lc)
    
    plt.plot(li50, nap50, 'o', markersize=8, color=c, markeredgecolor='k')
#    ax.autoscale()
plt.legend(result_names)
    
plt.yscale('log')
plt.xlabel('Leakage Index')
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

plt.ylabel('$N_{ap}$')
plt.xlim(0, 0.5)
plt.ylim(0, 10**9)

""" Boxplots of each variable """
plt.close('all')
#fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
fig, axs = plt.subplots(4, 2, sharex=False, sharey=False)
varsdo = ['Number of Alternative Paths', 'Leakage Index', 'Link Sharing Index', 
          'Flux Sharing Index', 'Topologic Mutual Information', 'Dynamic Mutual Information',
          'Topologic Conditional Entropy', 'Dynamic Conditional Entropy']

for v in varsdo:
    if v == 'Number of Alternative Paths':
        pp = [0,0]
        dkey = 'n_alt_paths'
        ylims = [1, 10**10]
    elif v == 'Leakage Index':
        pp = [0,1]
        dkey = 'leakage_idx'
        ylims = [0, .5]
    elif v == 'Link Sharing Index':
        pp = [1,0]
        dkey = 'top_link_sharing_idx'
        ylims = [0, 1]
    elif v == 'Flux Sharing Index':
        pp = [1,1]
        dkey = 'flux_sharing_idx'
        ylims = [0, 1]
    elif v == 'Topologic Mutual Information':
        pp = [2,0]
        dkey = 'top_mutual_info'
        ylims = [0, 5]
    elif v == 'Dynamic Mutual Information':
        pp = [2,1]
        dkey = 'dyn_mutual_info'
        ylims = [0, 5]
    elif v == 'Topologic Conditional Entropy':
        pp = [3,0]
        dkey = 'top_conditional_entropy'
        ylims = [0, 1.2]
    elif v == 'Dynamic Conditional Entropy':
        pp = [3,1]
        dkey = 'dyn_conditional_entropy'
        ylims = [0, 1.2]
        
        
    data = []    
    for i, (d, dp) in enumerate(zip(result_names, result_paths)):
        
        with open(dp, 'rb') as f:
            deltavars = pickle.load(f)
        
        tempdata = deltavars[dkey]
        if tempdata.shape[1] > 1:
            data.append(tempdata[:,1])
        else:
            data.append(tempdata)
    
    # Plot the data
    axs[pp[0],pp[1]].boxplot(data, labels=result_names, showfliers=False)
    axs[pp[0],pp[1]].set_xticklabels(axs[pp[0],pp[1]].get_xticklabels(), rotation=30)
    
    # Log scale for Nap
    if v == 'Number of Alternative Paths':
        axs[pp[0],pp[1]].set_yscale('log')
        
    # Set y axis limit
    axs[pp[0],pp[1]].set_ylim(ylims)
        
    # Add title
    axs[pp[0],pp[1]].set_title(v)


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)


""" Boxplots of each variable """
plt.close('all')
#fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
fig, axs = plt.subplots()

data = []
for i, (d, dp) in enumerate(zip(result_names, result_paths)):

    with open(dp, 'rb') as f:
        deltavars = pickle.load(f)

    tempdata = deltavars['resistance_distance']
    if tempdata.shape[1] > 1:
        data.append(tempdata[:,1])
    else:
        data.append(tempdata)

# Plot the data
axs.boxplot(data, labels=result_names, showfliers=False)
axs.set_xticklabels(axs.get_xticklabels(), rotation=30)
    
# Add title
axs.set_title('Resistance Distance')














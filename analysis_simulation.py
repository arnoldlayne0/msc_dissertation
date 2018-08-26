import os
os.chdir('C:/Users/ELITEBOOK840/Desktop/diss/code_public')

### this script is used to 
### a) run the simulations and summarise the results
###     1) empirical time complexity of PCRM IP and Top-k greedy algorithms
###     2) approximation quality of the greedy algorithm for PCRM
###     3) similarity of the results of PCRM and Top-k rankings corresponing to the same notions of fairness
###     4) price of fairness - the loss of utility in fair results 
### b) test the algorithms on a manufactured biased example

### do not run the simulations (unless you really wish to), use the provided results 
### from the simulation_results folder instead (read with the code provided in this script)

import math
import numpy as np
import pandas as pd
import simulation_setup as es
import data_gen as dg
import ranking_algorithms as ra
import matplotlib.pyplot as plt
#import importlib
#importlib.reload(es)

# global parameters for all simulations (type of constraints and number of runs)
notion = 'demographic'
times = 1000

##### Run all simulations #####
np.random.seed(123)
### empirical time complexity ip
df_pcrm = es.sim_pcrm(times, notion)
df_pcrm.to_csv('df_pcrm_time.txt')

### empirical time complexity topk
df_topk = es.sim_topk(times, notion)
df_topk.to_csv('df_topk_time.txt')

### approximation quality
df_greedy_qual_disjoint= es.greedy_approx(times, notion, True)
df_greedy_qual_disjoint.to_csv('df_greedy_qual_disjoint.txt')

df_greedy_qual_nodisj= es.greedy_approx(times, notion, False)
df_greedy_qual_nodisj.to_csv('df_greedy_qual_nodisj.txt')

### similarity
df_pcrm_vs_topk_dem = es.pcrm_vs_topk(times, notion)
df_pcrm_vs_topk_dem['p_bonferonni'] = df_pcrm_vs_topk_dem['p_val'] * times 
df_pcrm_vs_topk_dem.to_csv('df_pcrm_vs_topk_dem.txt')

### price of fairness
df_price = es.price_of_fairness(times, notion)
df_price.to_csv('df_price.txt')

df_price_util = es.price_of_fairness(times, 'utilitarian')
df_price_util.to_csv('df_price_util.txt')

##### EMPIRICAL TIME COMPLEXITY #####
### INTEGER PROGRAM PCRM ###
df_pcrm = pd.read_csv('df_pcrm_time.txt')

no_vars = df_pcrm.loc[:,'no_items'] * df_pcrm.loc[:,'no_pos']
no_constraints = df_pcrm.loc[:,'no_items'] + df_pcrm.loc[:,'no_pos'] + df_pcrm.loc[:,'no_prop']
var_cons = no_vars * no_constraints
time = df_pcrm.loc[:,'time']

df_sum = pd.concat([time, no_vars, no_constraints], axis=1)
df_sum.columns = ['time', 'variables', 'constraints']
df_sum.describe()

plt.subplot(121)
plt.scatter(df_sum.loc[:,'variables'], df_sum.loc[:,'time'])
plt.xlabel('Number of variables')
plt.ylabel('Time (s)')
plt.subplot(122)
plt.scatter(df_sum.loc[:,'constraints'], df_sum.loc[:,'time'])
plt.xlabel('Number of constraints')
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/iprunningtime.pdf')


### GREEDY TOPK ###
df_topk = pd.read_csv('df_topk_time.txt')

time_topk = df_topk.loc[:,'time']
df_sum_topk = df_topk[['time','no_items', 'no_pos']]
df_sum_topk.describe()

plt.subplot(121)
plt.scatter(df_topk.loc[:,'no_items'], df_topk.loc[:,'time'])
plt.xlabel('Number of items')
plt.ylabel('Time (s)')
plt.subplot(122)
plt.scatter(df_topk.loc[:,'no_pos'], df_topk.loc[:,'time'])
plt.xlabel('Number of positions')
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/greedytopkrunningtime.pdf')


###### GREEDY APPROX QUALITY #####
### disjoint properties
df_greedy_qual_disjoint = pd.read_csv('df_greedy_qual_disjoint.txt', index_col=0)
df_greedy_qual_disjoint.describe()

### non-disjoint properties
df_greedy_qual_nodisj = pd.read_csv('df_greedy_qual_nodisj.txt', index_col=0)
df_greedy_qual_nodisj.describe()


##### PCRM and TOPK similarity #####
df_pcrm_vs_topk_dem = pd.read_csv('df_pcrm_vs_topk_dem.txt', index_col=0)
df_pcrm_vs_topk_dem.describe()
(pd.to_numeric(df_pcrm_vs_topk_dem['p_bonferonni']) <= 0.5).sum()

##### PRICE OF FAIRNESS #####
from scipy.stats import pearsonr
df_price = pd.read_csv('df_price.txt', index_col=0)
df_price.describe()
pearsonr(df_price.loc[:,'price of fairness'].values, df_price.loc[:,'max_unfairness'].values)

df_price_util = pd.read_csv('df_price_util.txt', index_col=0)
df_price_util.describe()
pearsonr(df_price_util.loc[:,'price of fairness'].values, df_price.loc[:,'max_unfairness'].values)

##### Manufactured biased example #####
### IMBALANCED DEMOGRAPHY ###
# generate biased data
m,n,p = 10,5,2
news = np.arange(0,m)
side = np.zeros((m,p), dtype = int)
for item in range(0,n):
    side[item][0] = 1
for item in range(n,m):
    side[item][1] = 1
utilities = np.linspace(start = .99,stop = .9, num = m)
# generate position exposure according to DCG
exposure = np.zeros(n)
for j in range(n):
    exposure[j] = 1/(math.log(j+2))
side_list = dg.get_prop_list_sim(side)
val_mat = dg.get_val_mat(utilities, exposure)

# construct unconstrained, optimal PCRM, approx PCRM and Top-k rankings
unconstrained = ra.unconstrained_ranking_matching(val_mat)
unconstrained_rank_mat = ra.rank_mat_match(unconstrained[0],m,n)
unconstrained_rank_list = ra.rank_list_from_mat(unconstrained_rank_mat)

dem_parity = dg.parity_pcrm(side_list, utilities, 'demographic', 1.1)
fair = ra.ip_parity(utilities, exposure, side_list, dem_parity)
fair_rank_mat = ra.rank_mat_lp(fair[3],m,n)
fair_rank_list = ra.rank_list_from_mat(fair_rank_mat)
frm = np.zeros((m,n))
frm[:][0:5] = fair_rank_mat[:][0:5]
frm[:][5:10] = fair_rank_mat[:][5:10] * 1/2

fair_greedy = ra.greedy_parity_two(utilities, exposure, side, dem_parity)
fair_greedy_rank_mat = ra.rank_mat_greedy(fair_greedy[0],m,n)
fgrm = np.zeros((m,n))
fgrm[:][0:5] = fair_greedy_rank_mat[:][0:5]
fgrm[:][5:10] = fair_greedy_rank_mat[:][5:10] * 1/2

parity_topk = dg.parity_topk(side_list, utilities, n, 'demographic')
greedy_rank_topk = ra.greedy_topk(side, parity_topk, exposure, utilities)  
fair_greedy_topk_rank_mat = ra.rank_mat_greedy(greedy_rank_topk[0],m,n)
fair_greedy_topk_rank_mat_groups = ra.two_group_mat(fair_greedy_topk_rank_mat, side_list[1], m, n)       
fgtrm = np.zeros((m,n))
fgtrm[:][0:5] = fair_greedy_topk_rank_mat[:][0:5]
fgtrm[:][5:10] = fair_greedy_topk_rank_mat[:][5:10] * 1/2


# summarise the unfairness of the unconstrained result (demographic parity ratio) and the price of fair results
dpr = ra.dp_ratio(unconstrained_rank_list, exposure, side_list)
dpr_fair = ra.dt_ratio(fair_rank_list, exposure, utilities, side_list)
pof_exact =  (unconstrained[1] - fair[2])/(unconstrained[1])
pof_approx =  (unconstrained[1] - fair_greedy[1])/(unconstrained[1])
pof_topk = (unconstrained[1] - greedy_rank_topk[1])/(unconstrained[1])


# visualise the ranking matrices
font = {'family': 'serif','color':'white','weight': 'normal','size': 28}
font_two = {'family': 'serif','color':'black','weight': 'normal','size': 28}
fontA= {'family': 'serif','color':  'yellow', 'weight': 'normal','size': 28}
fontB = {'family': 'serif', 'color':  '#70CBFF', 'weight': 'normal','size': 28}

plt.matshow(unconstrained_rank_mat)
plt.ylabel('item', fontdict=font_two)
plt.xlabel('position', fontdict=font_two)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.text(0, 8.25, r'value = ' + str(np.round(unconstrained[1],3)), fontdict=font)
plt.text(0, 3.25, r'liberal', fontdict=fontA)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_unc.pdf')

plt.matshow(frm)
plt.xlabel('position', fontdict=font_two)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.text(0, 8.3, r'value = ' + str(np.round(fair[2],3)), fontdict=font)
plt.text(0, 2.25, r'liberal', fontdict=fontA)
plt.text(0, 4.25, r'conservative', fontdict=fontB)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_ip.pdf')

plt.matshow(fgrm)
plt.xlabel('position', fontdict=font_two)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.text(0, 8.3, r'value = ' + str(np.round(fair_greedy[1],3)), fontdict=font)
plt.text(0, 2.25, r'liberal', fontdict=fontA)
plt.text(0, 4.25, r'conservative', fontdict=fontB)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_greedy.pdf')

plt.matshow(fgtrm)
plt.xlabel('position', fontdict=font_two)
plt.text(0, 8.3, r'value = ' + str(np.round(greedy_rank_topk[1],3)), fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.text(0, 2.25, r'liberal', fontdict=fontA)
plt.text(0, 4.25, r'conservative', fontdict=fontB)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_topk.pdf')


### IMBALANCED UTILITY ###
#use the same data but disparate treatment constraints
# construct unconstrained (same as before), optimal PCRM, approx PCRM and Top-k rankings
dem_parity = dg.parity_pcrm(side_list, utilities, 'utilitarian', 1.2)
fair = ra.ip_parity(utilities, exposure, side_list, dem_parity)
fair_rank_mat = ra.rank_mat_lp(fair[3],m,n)
fair_rank_list = ra.rank_list_from_mat(fair_rank_mat)
frm = np.zeros((m,n))
frm[:][0:5] = fair_rank_mat[:][0:5]
frm[:][5:10] = fair_rank_mat[:][5:10] * 1/2

fair_greedy = ra.greedy_parity_two(utilities, exposure, side, dem_parity)
fair_greedy_rank_mat = ra.rank_mat_greedy(fair_greedy[0],m,n)
fgrm = np.zeros((m,n))
fgrm[:][0:5] = fair_greedy_rank_mat[:][0:5]
fgrm[:][5:10] = fair_greedy_rank_mat[:][5:10] * 1/2

parity_topk = dg.parity_topk(side_list, utilities, n, 'utilitarian')
greedy_rank_topk = ra.greedy_topk(side, parity_topk, exposure, utilities)  
fair_greedy_topk_rank_mat = ra.rank_mat_greedy(greedy_rank_topk[0],m,n)
fgtrm = np.zeros((m,n))
fgtrm[:][0:5] = fair_greedy_topk_rank_mat[:][0:5]
fgtrm[:][5:10] = fair_greedy_topk_rank_mat[:][5:10] * 1/2

# summarise the unfairness of the unconstrained result (disparate treatment ratio) and the price of fair results
dtr = ra.dt_ratio(unconstrained_rank_list, exposure, utilities, side_list)
dtr_fair = ra.dt_ratio(fair_rank_list, exposure, utilities, side_list)
pof_exact =  (unconstrained[1] - fair[2])/(unconstrained[1])
pof_approx =  (unconstrained[1] - fair_greedy[1])/(unconstrained[1])
pof_topk = (unconstrained[1] - greedy_rank_topk[1])/(unconstrained[1])


# visualise the ranking matrices
plt.matshow(unconstrained_rank_mat)
plt.ylabel('item', fontdict=font_two)
plt.xlabel('position', fontdict=font_two)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.text(0, 8.25, r'value = ' + str(np.round(unconstrained[1],3)), fontdict=font)
plt.text(0, 3.25, r'liberal', fontdict=fontA)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_unc_util.pdf')

plt.matshow(frm)
plt.xlabel('position', fontdict=font_two)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.text(0, 8.3, r'value = ' + str(np.round(fair[2],3)), fontdict=font)
plt.text(0, 2.25, r'liberal', fontdict=fontA)
plt.text(0, 4.25, r'conservative', fontdict=fontB)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_ip_util.pdf')

plt.matshow(fgrm)
plt.xlabel('position', fontdict=font_two)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.text(0, 8.3, r'value = ' + str(np.round(fair_greedy[1],3)), fontdict=font)
plt.text(0, 2.25, r'liberal', fontdict=fontA)
plt.text(0, 4.25, r'conservative', fontdict=fontB)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_greedy_util.pdf')

plt.matshow(fgtrm)
plt.xlabel('position', fontdict=font_two)
plt.text(0, 8.3, r'value = ' + str(np.round(greedy_rank_topk[1],3)), fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
plt.text(0, 2.25, r'liberal', fontdict=fontA)
plt.text(0, 4.25, r'conservative', fontdict=fontB)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/synth_topk_util.pdf')


### TWO-GROUP SIMULATION ANALYSIS ###
np.random.seed(123)
m , n, times, notion, viol = 30, 20, 1000, 'demographic', 1.2
df_twogroup_dem, one_details_dem, two_details_dem = es.twogroup_sim_mult_uniform(m, n, times, notion, viol)
df_twogroup_dem.to_csv('df_twogroup_dem.txt')
df_twogroup_dem = pd.read_csv('df_twogroup_dem.txt', index_col=0)
df_twogroup_dem.describe()
from scipy.stats import pearsonr
pearsonr(df_twogroup_dem.loc[:,'price'].values, df_twogroup_dem.loc[:,'max_unfair_unfair'].values)



m , n, times, notion, viol = 30, 20, 1000, 'utilitarian', 1.2
df_twogroup_util, one_details_util, two_details_util = es.twogroup_sim_mult_uniform(m, n, times, notion, viol)
df_twogroup_util.to_csv('df_twogroup_util.txt')
df_twogroup_util = pd.read_csv('df_twogroup_util.txt', index_col=0)
df_twogroup_util.describe()
from scipy.stats import pearsonr
pearsonr(df_twogroup_util.loc[:,'price'].values, df_twogroup_util.loc[:,'max_unfair_unfair'].values)







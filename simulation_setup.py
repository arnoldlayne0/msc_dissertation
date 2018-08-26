# this script provides functions used for repeatedly solving the ranking problems
# and summarising the results to study the empirical behaviour of the algorithms
# ie their time complexity, approximation quality, price of the fair solutions

import numpy as np
import pandas as pd
import ranking_algorithms as ra
import data_gen as dg
import math
from scipy import stats
import timeit
#import importlib
#importlib.reload(ra)

def sim_pcrm(times, notion):
    """ solves pcrm times times on randomly generated data
    Returns the time (in seconds) that solving each random instance
    of the problem took as well as the size of the instances
    """
    time, value, bound, no_items, no_pos, no_prop = [], [], [], [], [],[]

    for i in range(times):
        p = np.random.randint(low = 2, high = 20, size = 1)[0]
        n = np.random.randint(low = 3, high = 100, size = 1)[0]
        m = math.ceil(np.random.uniform(low=1,high=1.5,size=1) * n)
        viol = 1.5
        #viol = np.random.uniform(low = 1.2, high = 2.0, size = 1)[0]
        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list_sim(properties)
        parity_pcrm = dg.parity_pcrm(prop_list, item_qual, notion, viol)
        start = timeit.default_timer()
        rank_pcrm = ra.ip_parity(item_qual, pos_imp, prop_list, parity_pcrm)
        stop = timeit.default_timer()
        elapsed = (stop - start)
        time.append(elapsed)
        value.append(rank_pcrm[2])
        bound.append(viol)
        no_items.append(m)
        no_pos.append(n)
        no_prop.append(p)

    # use bonferonni for multiple comparisons        
    dat = np.array([time, value, bound, no_items, no_pos, no_prop]).T        
    df = pd.DataFrame(dat, columns = ['time', 'value', 'bound', 'no_items', 'no_pos', 'no_prop'])
    
    return df

def sim_topk(times, notion):
    """ solves topk rm times times on randomly generated data
    Returns the time (in seconds) that solving each random instance
    of the problem took as well as the size of the instances
    """
    time, value, bound, no_items, no_pos, no_prop = [], [], [], [], [], []
    for i in range(times):
        p = np.random.randint(low = 2, high = 20, size = 1)[0]
        n = np.random.randint(low = 3, high = 100, size = 1)[0]
        m = math.ceil(np.random.uniform(low=1,high=1.5,size=1) * n)
        viol = 1.5
        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list_sim(properties)
        parity_topk = dg.parity_topk(prop_list, item_qual, n, notion)
        start = timeit.default_timer()
        rank_topk = ra.greedy_topk(properties, parity_topk, pos_imp, item_qual)
        stop = timeit.default_timer()
        elapsed = (stop - start)
        time.append(elapsed)
        value.append(rank_topk[1])
        bound.append(viol)
        no_items.append(m)
        no_pos.append(n)
        no_prop.append(p)
            
    dat = np.array([time, value, bound, no_items, no_pos, no_prop]).T        
    df = pd.DataFrame(dat, columns = ['time', 'value', 'bound', 'no_items', 'no_pos', 'no_prop'])

    return df

def pcrm_vs_topk(times, notion):
    """ solves pcrm and topk rm times times on randomly generated data
    Returns the correlation between the solutions to the two problems
    and the information about the instances
    """
    value_factor, corr, sign, no_items, no_pos, no_prop = [], [], [], [], [], []
    for i in range(times):
        p = np.random.randint(low = 2, high = 5, size = 1)[0]
        n = np.random.randint(low = 10, high = 20, size = 1)[0]
        m = math.ceil(np.random.uniform(low=1,high=1.5,size=1) * n)
        viol = np.random.uniform(low = 1.1, high = 1.2, size = 1)[0]
        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list_sim(properties)
        parity_topk = dg.parity_topk(prop_list, item_qual, n, notion)
        parity_pcrm = dg.parity_pcrm(prop_list, item_qual, notion, viol)   
        rank_topk = ra.greedy_topk(properties, parity_topk, pos_imp, item_qual)
        rank_pcrm = ra.ip_parity(item_qual, pos_imp, prop_list, parity_pcrm)
        rank_list_topk, val_topk = rank_topk[0], rank_topk[1]
        rank_list_pcrm, val_pcrm = rank_pcrm[3], rank_pcrm[2]
        rank_list_pcrm = ra.rank_mat_lp(rank_list_pcrm, m, n)
        rank_list_pcrm = ra.rank_list_from_mat(rank_list_pcrm)
        # return difference in value + spearman rho and maybe size
        rho, p_val = stats.spearmanr(rank_list_topk, rank_list_pcrm)
        if val_topk != 0:
            value_factor.append(val_pcrm/val_topk)
        elif val_pcrm == 0:
            value_factor.append(1)
        else:
            value_factor.append(0)
        corr.append(rho)
        sign.append(p_val)
        no_items.append(m)
        no_pos.append(n)
        no_prop.append(p)
        
        print(str(i) + ' pcrm_vs_topk completed')

    
    dat = np.array([value_factor, corr, sign, no_items, no_pos, no_prop]).T
    df = pd.DataFrame(dat, columns = ['value_factor', 'rho', 'p_val', 'no_items', 'no_pos', 'no_prop'])

    
    return df

def greedy_approx(times, notion, disjoint):
    '''Solves PCRM using the greedy approximation algorithm
    Returns the approximation factor (as compared to the exact solution from the IP)
    whether the instance was feasible and if the greedy algorithm found a feasible solution'''

    value_factor, feas_ip, feas_greedy, time_ip, time_greedy, viol_coeff = [], [], [], [], [], []
    for i in range(times):
        p = np.random.randint(low = 2, high = 5, size = 1)[0]
        n = np.random.randint(low = 10, high = 20, size = 1)[0]
        m = math.ceil(np.random.uniform(low=1,high=1.5,size=1) * n)
        viol = np.random.uniform(low = 1.1, high = 1.2, size = 1)[0]
        pos_imp, item_qual, properties = dg.sim_data(m, n, p, disjoint)
        prop_list = dg.get_prop_list_sim(properties)
        parity_pcrm = dg.parity_pcrm(prop_list, item_qual, notion, viol)   
        
        start = timeit.default_timer()
        rank_ip = ra.ip_parity(item_qual, pos_imp, prop_list, parity_pcrm)
        stop = timeit.default_timer()
        elapsed_ip = (stop - start)
        start = timeit.default_timer()
        rank_greedy = ra.greedy_parity(item_qual, pos_imp, properties, parity_pcrm)
        stop = timeit.default_timer()
        elapsed_greedy = (stop - start)
        if rank_ip[2] > 0:
            f_ip = 1
        else:
            f_ip = 0
        if rank_greedy[1] > 0:
            f_greedy = 1
        else:
            f_greedy = 0
        if f_ip == 1 and f_greedy == 1:
            val_fact = rank_ip[2]/rank_greedy[1]
        else:
            val_fact = np.nan
        
        value_factor.append(val_fact)
        feas_ip.append(f_ip)
        feas_greedy.append(f_greedy)
        time_ip.append(elapsed_ip)
        time_greedy.append(elapsed_greedy)
        viol_coeff.append(viol)
        
        print(str(i) + ' greedy_approx completed')
            
    dat = np.array([value_factor, feas_ip, feas_greedy, time_ip, time_greedy, viol_coeff]).T
    df = pd.DataFrame(dat, columns = ['value_factor', 'feas_ip', 'feas_greed', 'time_ip', 'time_greedy', 'viol_coeff'])

    return df

def price_of_fairness(times, notion):
    ''' solves the pcrm and compares the value of the fair ranking to the value of the unconstrained solution,
    returns the price of fairness and maximum unfairness of the unconstrained ranking
    as well as information on the instances'''
    
    
    price_fair, viol_coeff, unfairness, no_items, no_pos, no_prop = [], [], [], [], [], []

    for i in range(times):
        p = np.random.randint(low = 2, high = 5, size = 1)[0]
        n = np.random.randint(low = 10, high = 20, size = 1)[0]
        m = math.ceil(np.random.uniform(low=1,high=1.5,size=1) * n)
        viol = np.random.uniform(low = 1.15, high = 1.2, size = 1)[0]
        #viol = np.random.uniform(low = 1.2, high = 2.0, size = 1)[0]
        pos_imp, item_qual, properties = dg.sim_data(m, n, p, True)
        prop_list = dg.get_prop_list_sim(properties)
        parity_pcrm = dg.parity_pcrm(prop_list, item_qual, notion, viol)
        rank_pcrm = ra.ip_parity(item_qual, pos_imp, prop_list, parity_pcrm)
        fair = rank_pcrm[2]
        val_mat = dg.get_val_mat(pos_imp, item_qual)

        rank_unc = ra.unconstrained_ranking_matching(val_mat)
        opt = rank_unc[1]
        unc_rank_mat = ra.rank_mat_match(rank_unc[0],m,n)
        unc_rank_list = ra.rank_list_from_mat(unc_rank_mat)
        unfair = ra.max_unfair(unc_rank_list, pos_imp, parity_pcrm/viol, prop_list, viol)
        price = (opt-fair)/opt
        price_fair.append(price)
        viol_coeff.append(viol)
        unfairness.append(unfair)
        no_items.append(m)
        no_pos.append(n)
        no_prop.append(p)

        print(str(i) + ' price of fairness completed')


    # use bonferonni for multiple comparisons        
    dat = np.array([price_fair, viol_coeff, unfairness, no_items, no_pos, no_prop]).T        
    df = pd.DataFrame(dat, columns = ['price of fairness', 'viol_coeff', 'max_unfairness', 'no_items', 'no_pos', 'no_prop'])
    
    return df

###### TWO-GROUP DATA #####

# this part contains a set of functions to generate data with items whose 
# utilities and properties come from two different distrubitions

import scipy.stats

def sim_twogroup_data(one, two):
    '''one = [mean, sd, size, properies_prob]
       two = [mean, sd, size, properies_prob]
       properties_prob = [p1, p2]
       notion = 'demographic', 'utilitarian'
    '''
    m = one[2] + two[2]
    p = len(one[3])
    lower, upper = 0, 1
    sigma = 0.1
    item_qual_1 = scipy.stats.truncnorm.rvs(
          (lower-one[0])/sigma,(upper-one[0])/sigma,loc=one[0],scale=one[1],size=one[2])
    item_qual_2 = scipy.stats.truncnorm.rvs(
          (lower-two[0])/sigma,(upper-two[0])/sigma,loc=two[0],scale=two[1],size=two[2])
    prop_like_1 = one[3]
    prop_like_2 = two[3]
    properties = np.zeros((m,p), dtype = int)
    for item in range(0,one[2]):
        for l in range(0,p):
            properties[item][l] = np.random.binomial(1,prop_like_1[l],1)
        for item in range(one[2],m):
            for l in range(0,p):
                properties[item][l] = np.random.binomial(1,prop_like_2[l],1)
    item_qual = np.append([item_qual_1], [item_qual_2])
    dat = pd.DataFrame(properties)
    dat['i'] = item_qual
    dat = dat.sort_values(by = 'i', ascending=False)
    dat = dat.reset_index(drop=True)    
    item_qual = dat['i'].values
    properties = dat.loc[:,dat.columns!='i'].values
    prop_list = dg.get_prop_list_sim(properties)


    return item_qual, prop_list, properties



def twogroup_sim(one, two, n, notion, viol):
     
    item_qual, prop_list, properties = sim_twogroup_data(one, two)
    m = len(item_qual)
    exposure = np.zeros(n)
    for j in range(n):
        exposure[j] = 1/(math.log(j+2))

    val_mat = dg.get_val_mat(item_qual, exposure)
    unconstrained = ra.unconstrained_ranking_matching(val_mat)
    unconstrained_rank_list = np.arange(0,n)
    unc_val = unconstrained[1]
        
       
    dem_parity = dg.parity_pcrm(prop_list, item_qual, notion, viol)
    fair = ra.ip_parity(item_qual, exposure, prop_list, dem_parity)
    fair_rank_mat = ra.rank_mat_lp(fair[3],m,n)
    fair_rank_list = ra.rank_list_from_mat(fair_rank_mat)
    fair_val = fair[2]
    
    fair_approx = ra.greedy_parity_two(item_qual, exposure, properties, dem_parity)
    if fair_approx[1] != 0:
        val_fact = fair_val/fair_approx[1]
    else:
        val_fact = np.nan

    if notion == 'demographic':
        r_unfair = ra.dp_ratio(unconstrained_rank_list, exposure, prop_list)
        r_fair = ra.dp_ratio(fair_rank_list, exposure, prop_list)
    elif notion =='utilitarian':
        r_unfair = ra.dt_ratio(unconstrained_rank_list, exposure, item_qual, prop_list)
        r_fair = ra.dt_ratio(fair_rank_list, exposure, item_qual, prop_list)

    mu_unfair = ra.max_unfair(unconstrained_rank_list, exposure, dem_parity, prop_list, viol)
    mu_fair = ra.max_unfair(fair_rank_list, exposure, dem_parity, prop_list, viol)
    if unc_val == 0:
        pof = 0
    else:
        pof = (unc_val - fair_val)/unc_val

    
    return pof, r_unfair, r_fair, mu_unfair, mu_fair, one, two, val_fact

       
def twogroup_sim_mult(o, t, m, n, times, notion, viol):
     
    price = []
    ratio_fair = []
    ratio_unfair = []
    max_unfair_fair = []
    max_unfair_unfair = []
    one_details = []
    two_details = []
    approx_quality = []
    for i in range(times):
        mean_one = np.random.normal(o[0], 0.1, 1)[0]
        sd_one = abs(np.random.normal(o[1], 0.1, 1)[0])
        size_one = np.random.randint(0,m)
        prob_one = [o[2], o[3]]
        mean_two = np.random.normal(t[0], 0.1, 1)[0]
        sd_two = abs(np.random.normal(t[1], 0.1, 1)[0])
        size_two = m - size_one
        prob_two = [t[2], t[3]]

        one = [mean_one, sd_one, size_one, prob_one]
        two = [mean_two, sd_two, size_two, prob_two]
        results = twogroup_sim(one, two, n, notion, viol)

        price.append(results[0])
        ratio_unfair.append(results[1])
        ratio_fair.append(results[2])
        max_unfair_unfair.append(results[3])
        max_unfair_fair.append(results[4])
        one_details.append(results[5])
        two_details.append(results[6])
        approx_quality.append(results[7])
        
        print('iteration ' + str(i) + ' done')  
        
    
    dat = np.array([price, ratio_unfair, ratio_fair, max_unfair_unfair, max_unfair_fair, approx_quality]).T        
    df = pd.DataFrame(dat, columns = ['price', 'ratio_unfair', 'ratio_fair', 'max_unfair_unfair',
                                      'max_unfair_fair', 'approx_quality'])
    
    return df, one_details, two_details

def twogroup_sim_mult_uniform(m, n, times, notion, viol):
     
    price = []
    ratio_fair = []
    ratio_unfair = []
    max_unfair_fair = []
    max_unfair_unfair = []
    one_details = []
    two_details = []
    approx_quality = []
    for i in range(times):
        mean_one = np.random.uniform(0.6, 0.8, 1)[0]
        sd_one = np.random.uniform(0.1, 0.3, 1)[0]
        size_one = np.random.randint(0,m)
        prob_one = [0.8, 0.2]
        mean_two = np.random.uniform(0.2, 0.4, 1)[0]
        sd_two = np.random.uniform(0.1, 0.3, 1)[0]
        size_two = m - size_one
        prob_two = [0.2, 0.8]

        one = [mean_one, sd_one, size_one, prob_one]
        two = [mean_two, sd_two, size_two, prob_two]
        results = twogroup_sim(one, two, n, notion, viol)

        price.append(results[0])
        ratio_unfair.append(results[1])
        ratio_fair.append(results[2])
        max_unfair_unfair.append(results[3])
        max_unfair_fair.append(results[4])
        one_details.append(results[5])
        two_details.append(results[6])
        approx_quality.append(results[7])
        
        print('iteration ' + str(i) + ' done')  
        
    
    dat = np.array([price, ratio_unfair, ratio_fair, max_unfair_unfair, max_unfair_fair, approx_quality]).T        
    df = pd.DataFrame(dat, columns = ['price', 'ratio_unfair', 'ratio_fair', 'max_unfair_unfair',
                                      'max_unfair_fair', 'approx_quality'])
    
    return df, one_details, two_details



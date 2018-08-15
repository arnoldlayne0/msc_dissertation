# this scrip provides functions used for repeatedly solving the ranking problems
# and summarising the results to study the empirical behaviour of the algorithms
# ie their time complexity, approximation quality, price of the fair solutions

import numpy as np
import pandas as pd
import ranking_algorithms as ra
import setup as st
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
        pos_imp, item_qual, properties = st.sim_data(m, n, p, True)
        prop_list = st.get_prop_list_sim(properties)
        parity_pcrm = ra.parity_pcrm(prop_list, item_qual, notion, viol)
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
        pos_imp, item_qual, properties = st.sim_data(m, n, p, True)
        prop_list = st.get_prop_list_sim(properties)
        parity_topk = ra.parity_topk(prop_list, item_qual, n, notion, viol)
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
        pos_imp, item_qual, properties = st.sim_data(m, n, p, True)
        prop_list = st.get_prop_list_sim(properties)
        parity_topk = ra.parity_topk(prop_list, item_qual, n, notion, viol)
        parity_pcrm = ra.parity_pcrm(prop_list, item_qual, notion, viol)   
        rank_topk = ra.greedy_topk(properties, parity_topk, pos_imp, item_qual)
        rank_pcrm = ra.ip_parity(item_qual, pos_imp, prop_list, parity_pcrm)
        rank_list_topk, val_topk = rank_topk[0], rank_topk[1]
        rank_list_pcrm, val_pcrm = rank_pcrm[3], rank_pcrm[2]
        rank_list_pcrm = ra.rank_mat_lp(rank_list_pcrm, m, n)
        rank_list_pcrm = ra.rank_list_from_mat(rank_list_pcrm)
        # return difference in value + spearman rho and maybe size
        rho, p_val = stats.spearmanr(rank_list_topk, rank_list_pcrm)
        value_factor.append(val_pcrm/val_topk)
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
        pos_imp, item_qual, properties = st.sim_data(m, n, p, disjoint)
        prop_list = st.get_prop_list_sim(properties)
        parity_pcrm = ra.parity_pcrm(prop_list, item_qual, notion, viol)   
        
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
        viol = np.random.uniform(low = 1.1, high = 1.2, size = 1)[0]
        #viol = np.random.uniform(low = 1.2, high = 2.0, size = 1)[0]
        pos_imp, item_qual, properties = st.sim_data(m, n, p, True)
        prop_list = st.get_prop_list_sim(properties)
        parity_pcrm = ra.parity_pcrm(prop_list, item_qual, notion, viol)
        rank_pcrm = ra.ip_parity(item_qual, pos_imp, prop_list, parity_pcrm)
        fair = rank_pcrm[2]
        val_mat = st.get_val_mat(pos_imp, item_qual)

        rank_unc = ra.unconstrained_ranking_matching(val_mat)
        opt = rank_unc[1]
        unc_rank_mat = ra.rank_mat_match(rank_unc[0],m,n)
        unc_rank_list = ra.rank_list_from_mat(unc_rank_mat)
        unfair = ra.max_unfair(unc_rank_list, pos_imp, parity_pcrm, prop_list)
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




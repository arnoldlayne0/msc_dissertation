import os
os.chdir('C:/Users/ELITEBOOK840/Desktop/diss/code_public')

from ortools.linear_solver import pywraplp

import numpy as np
import pandas as pd
import networkx as nx
import data_gen as dg


##### TOP-K CONSTRAINED RANKING MAXIMISATION #####
### Greedy algorithm
def greedy_topk(properties, up_bounds, pos_imp, item_qual):
    
    """ Implements a greedy algorithm for ranking items subject to fairness constraintes
    
    Solves the constrained ranking maximisation problem with constraints defined as 
    bounds on the number of items with a property at top k positions
    Works for the case with delta (cardinality of the biggest type) = 1 (any number of disjoint properties)
    
    Assumptions:
    -------------------
        1) items are sorted in decreasing order wrt their quality
        2) Monge property holds
    
    Arguments:
    ------------------
        1) properties = 1/0 vector with properties[i] = 1 if item i has the property
        2) up_bounds = integer matrix with up_bounds[p][k] = j 
                       if there can be at most j items with the property p at top j positions
        3) pos_imp = vector of position importance (eg fractions of users who examine items at a given position)
        4) item_qual = vector of item qualities (eq relevance to the query)
    
    """
   
    if type(properties) == pd.core.frame.DataFrame:
        properties = properties.as_matrix()
    no_pos = len(pos_imp)
    no_items = len(item_qual)
    ranking = []
    up_bounds = np.array(up_bounds).T
    curr_sat = np.zeros((len(properties[0])))
    for pos in range(no_pos):
        considered = 0
        while (len(ranking) <= pos and considered < no_items):
            if considered in ranking:
                considered += 1
            elif (curr_sat + properties[considered] <= up_bounds[pos]).all():
                ranking.append(considered)
                curr_sat += properties[considered]
            elif considered <= no_items:
                considered += 1
            else:
                ranking = "infeasible"              
                
    total_value = 0
    if len(ranking) < no_pos:
        ranking = "infeasible"
    else:
        for i in range(len(ranking)):
            total_value += pos_imp[i] * item_qual[ranking[i]]
    return ranking, total_value

##### PARITY-CONSTRAINED RANKING MAXIMISATION ######
### Integer programming exact algorithm ###
def ip_parity(item_qual, pos_imp, prop_list, parity):
    
    """ Solves the parity-constrained ranking maximisation problem as an Integer Program
    
    Arguments:
    ------------------
        1) item_qual = vector of item qualities (eq relevance to the query)
        2) pos_imp = vector of position importance (eg fractions of users who examine items at a given position)
        3) parity = vector of max fractions of exposure that can be assigned to a given group of items
        4) prop-list = set of sets of items that posses a given property (groups)
   
    """
    
    solver = pywraplp.Solver('SolveIntegerProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    m, n, p = len(item_qual), len(pos_imp), len(prop_list)
    val_list, bounds, A = dg.gen_data_lp(item_qual, pos_imp, prop_list, parity)
    
    ranking = [0] * (m*n)
    
    objective = solver.Objective()
    for i in range(0, m*n):
    # create variables - food[i] is the amount of money spent on foodstuff i
        ranking[i] = solver.IntVar(0.0, solver.infinity(), val_list[i][0])
        objective.SetCoefficient(ranking[i], val_list[i][1])
    objective.SetMaximization()

    constraints = [0] * len(bounds)
    for i in range(0, m):
        constraints[i] = solver.Constraint(-solver.infinity(), bounds[i][1])
        for j in range(0, m * n):
            constraints[i].SetCoefficient(ranking[j], A[i][j])
    for i in range(m, m + n):
        constraints[i] = solver.Constraint(bounds[i][1], bounds[i][1])
        for j in range(0, m * n):
            constraints[i].SetCoefficient(ranking[j], A[i][j])
    for i in range(m + n, m + n + p):
        constraints[i] = solver.Constraint(-solver.infinity(), bounds[i][1])
        for j in range(0, m * n):
            constraints[i].SetCoefficient(ranking[j], A[i][j])


    result_status = solver.Solve()
    
    ip_success = (result_status == solver.OPTIMAL)
    solution = []
    for variable in ranking:
        solution.append(variable.solution_value())
    feasible = sum(solution) >= n    
    obj_value = solver.Objective().Value()
    no_vars, no_cons = solver.NumVariables(), solver.NumConstraints()
    desc_sol = []
    for variable in ranking:
        desc_sol.append([variable.name(), variable.solution_value()])

    
    return ip_success, feasible, obj_value, solution, desc_sol, no_vars, no_cons


### Linear relaxation ###
def lp_parity(item_qual, pos_imp, prop_list, parity):
    
    """ Solves the LP relaxation of the parity-constrained ranking maximisation problem   
    
    Arguments:
    ------------------
        1) item_qual = vector of item qualities (eq relevance to the query)
        2) pos_imp = vector of position importance (eg fractions of users who examine items at a given position)
        3) parity = vector of max fractions of exposure that can be assigned to a given group of items
        4) prop-list = set of sets of items that posses a given property (groups)
      
    """
   
    solver = pywraplp.Solver('SolveIntegerProblem',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    
    m, n, p = len(item_qual), len(pos_imp), len(prop_list)
    val_list, bounds, A = dg.gen_data_lp(item_qual, pos_imp, prop_list, parity)
    
    ranking = [0] * (m*n)
    
    objective = solver.Objective()
    for i in range(0, m*n):
    # create variables - food[i] is the amount of money spent on foodstuff i
        ranking[i] = solver.NumVar(0.0, 1, val_list[i][0])
        objective.SetCoefficient(ranking[i], val_list[i][1])
    objective.SetMaximization()

    constraints = [0] * len(bounds)
    for i in range(0, m):
        constraints[i] = solver.Constraint(-solver.infinity(), bounds[i][1])
        for j in range(0, m * n):
            constraints[i].SetCoefficient(ranking[j], A[i][j])
    for i in range(m, m + n):
        constraints[i] = solver.Constraint(bounds[i][1], bounds[i][1])
        for j in range(0, m * n):
            constraints[i].SetCoefficient(ranking[j], A[i][j])
    for i in range(m + n, m + n + p):
        constraints[i] = solver.Constraint(-solver.infinity(), bounds[i][1])
        for j in range(0, m * n):
            constraints[i].SetCoefficient(ranking[j], A[i][j])

    result_status = solver.Solve()
    
    lp_success = (result_status == solver.OPTIMAL)
    solution = []
    for variable in ranking:
        solution.append(variable.solution_value())
    feasible = sum(solution) >= n    
    obj_value = solver.Objective().Value()
    no_vars, no_cons = solver.NumVariables(), solver.NumConstraints()
    desc_sol = []
    for variable in ranking:
        desc_sol.append([variable.name(), variable.solution_value()])

    
    return lp_success, feasible, obj_value, solution, desc_sol, no_vars, no_cons

### Greedy approximation algorithm ###
def greedy_parity(item_qual, pos_imp, properties, parity): 
    """ Implements a greedy algorithm for ranking items subject to fairness constraintes
    
    Gives an approximation to pcrm, works when there is a feasibility guarantee
    
    Arguments:
    ------------------
        1) properties = 1/0 vector with properties[i] = 1 if item i has the property
        2) parity (in (0,1>) = "demographic"/"utilitarian" 
        3) pos_imp = vector of position importance (eg fractions of users who examine items at a given position)
        4) item_qual = vector of item qualities (eq relevance to the query)
    
    """
    # generate general info needed for all other
    no_pos = len(pos_imp)
    no_items = len(item_qual)
    
    # use the greedy procedure to compute the ranking
    total_expo = sum(pos_imp)
    ranking = [-float('inf')] * no_pos
    curr_sat = np.zeros((len(properties[0])))
    for pos in range(no_pos):
        considered = 0
        while (ranking[pos] == -float('inf')):
            if considered >= no_items:
                ranking[pos] = float('inf')
            elif considered in ranking:
                 considered += 1
            elif (curr_sat + properties[considered] * pos_imp[pos] <= parity * total_expo).all():
                ranking[pos] = considered
                curr_sat += properties[considered] * pos_imp[pos]
            else:
                considered += 1

    if float('inf') in ranking:
        ranking = "infeasible"
        total_value = 0
    else:
        total_value = 0
        for i in range(len(ranking)):
            total_value += pos_imp[i] * item_qual[ranking[i]]

    return ranking, total_value

def greedy_parity_two(item_qual, pos_imp, properties, parity): 
    """ Implements a greedy algorithm for ranking items subject to fairness constraintes
    
    Gives an approximation to pcrm, works when there is a feasibility guarantee
    
    This is an implementation with a break in the inner loop instead of a conditional loop
    which proves to be faster in some experiments
    
    Arguments:
    ------------------
        1) properties = 1/0 vector with properties[i] = 1 if item i has the property
        2) parity (in (0,1>) = "demographic"/"utilitarian" 
        3) pos_imp = vector of position importance (eg fractions of users who examine items at a given position)
        4) item_qual = vector of item qualities (eq relevance to the query)
    
    """
    # generate general info needed for all other
    no_pos = len(pos_imp)
    no_items = len(item_qual)
    if type(properties) == pd.core.frame.DataFrame:
        properties = properties.as_matrix()
    # use the greedy procedure to compute the ranking
    total_expo = sum(pos_imp)
    ranking = [-float('inf')] * no_pos
    curr_sat = np.zeros((len(properties[0])))
    item_list = np.arange(no_items)
    for pos in range(no_pos):
        for i in range(len(item_list)):
            it = item_list[i]
            if (curr_sat + properties[it] * pos_imp[pos] <= parity * total_expo).all():
                ranking[pos] = it
                curr_sat += properties[it] * pos_imp[pos]
                item_list = np.delete(item_list, i)
                break
                
    if -float('inf') in ranking:
        ranking = "infeasible"
        total_value = 0
    else:
        total_value = 0
        for i in range(len(ranking)):
            total_value += pos_imp[i] * item_qual[ranking[i]]

    return ranking, total_value


##### Unconstrained ######
def unconstrained_ranking_matching(val_mat):
    G = nx.Graph()
    m = len(val_mat[0])
    n = len(val_mat)
    nodes = np.arange(0,n + m, 1)
    G.add_nodes_from(nodes)
    for pos in range(n):
        for item in range(n, n + m):
            G.add_edge(pos, item, weight = val_mat[pos][item - n])
    ranking = list(nx.max_weight_matching(G))
    
    for i in range(len(ranking)):
        ranking[i] = sorted(ranking[i])
    
    value = 0
    for i in range(len(ranking)):
        pos = ranking[i][0]
        item = ranking[i][1] - n
        value += val_mat[pos][item]
        
    return ranking, value

def easy_unc_rank(item_qual, pos_imp):
    n = len(pos_imp)
    rank_unc_val = 0
    rank_unc = np.zeros((n,2),dtype = int)
    for i in range(n):
        rank_unc[i]=[i,i]
        rank_unc_val += item_qual[i] * pos_imp[i]
    return rank_unc, rank_unc_val


##### Auxiliary functions ######
### transforming the output of the algorithms into some universal formats
def rank_mat_lp(rank_list,m,n):
    rank_mat = np.reshape(rank_list, (m, n))
    return rank_mat
    
def rank_mat_greedy(rank_list, m, n):
    rank_mat = np.zeros((m,n),dtype=int)
    for i in range(n):
        rank_list[i]
        rank_mat[rank_list[i]][i] = 1 
    return rank_mat

def rank_mat_match(rank_list,m,n):
    item = 0
    pos = 0
    rank_mat = np.zeros((m,n), dtype = int)
    for i in range(len(rank_list)):
        item = rank_list[i][0]
        pos = rank_list[i][1] - n
        rank_mat[item][pos] = 1
    return rank_mat

def rank_list_from_mat(rank_mat):
    m, n = rank_mat.shape
    rank_list = np.zeros(n, dtype = int)
    for i in range(m):
        for j in range(n):
            if rank_mat[i][j] == 1:
                rank_list[j] = i
    return rank_list

def two_group_mat(mat, prop_list, m, n):
    mat_two = mat.copy()
    for i in range(m):
        for j in range(n):
            if mat[i][j] == 1 and i in prop_list:
                mat_two[i][j] = 0.5
                
    return mat_two

def two_group_mat_two(mat, prop_list, m ,n):
    mat_two = mat.copy()
    for i in prop_list:
        mat_two[i] = mat[i] / 2
    return mat_two

### unfairness measures
def max_unfair(rank_list, pos_imp, parity, prop_list, viol):
    '''returns the largest factor of violation of fairness constraints
    in a given ranking '''
    p = len(prop_list)
    group_exp = np.zeros(p)
    total_exp = sum(pos_imp)        
    for i in range(len(rank_list)):
        for l in range(p):
            if rank_list[i] in prop_list[l]:
                group_exp[l] += pos_imp[i]/total_exp
    unfair = max(group_exp / (parity/viol + np.linspace(.001,.001,p)))
    
    return unfair

def dp_ratio(rank_list, pos_imp, prop_list):
    '''returns the demographic parity ratio between two groups'''
    p = len(prop_list)
    group_exp = np.zeros(p)        
    for i in range(len(rank_list)):
        for l in range(p):
            card = len(prop_list[l])
            if rank_list[i] in prop_list[l]:
                group_exp[l] += pos_imp[i]/card
    if group_exp[1] == 0:
        dpr = float('inf')
    else:
        dpr = group_exp[0]/group_exp[1]
    
    return dpr

def dt_ratio(rank_list, pos_imp, item_qual, prop_list):
    '''returns the disparate treatment ratio between two groups'''
    p = len(prop_list)
    group_exp = np.zeros(p)  
    group_util = np.zeros(p)
    for l in range(p):
        for i in prop_list[l]:
            group_util[l] += item_qual[i]
    for i in range(len(rank_list)):
        for l in range(p):
            if rank_list[i] in prop_list[l]:
                group_exp[l] += pos_imp[i]/group_util[l]
    if group_exp[1] == 0:
        dtr = float('inf')
    else:
        dtr = group_exp[0]/group_exp[1]
    
    return dtr



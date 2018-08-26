import os
os.chdir('C:/Users/ELITEBOOK840/Desktop/diss/code_public')

###### GENERATING DATA #####

# this script contains a set of functions to generate and manipulate data
# both random data and the real-world news recommendation dataset

import numpy as np
import pandas as pd
import math

### Simulated data ###
# m = number of items
# n = number of positions
# p = number of properties

def gen_distinct_properties(m,p):
    ''' generates p random distinct properties for m items'''
    properties = np.zeros((m,p), dtype = int)
    for item in range(0,m):
        prop = np.random.randint(low = 0, high = p)
        properties[item][prop] = 1
    return properties


def gen_properties(m,p):
    ''' generates p random properties for m items'''
    properties = np.zeros((m,p), dtype = int)
    for item in range(0,m):
        properties[item] = np.random.binomial(1,.5,p)
    return properties


def sim_data(m, n, p, distinct):
    '''generates random utilities of items, exposure of positions and properties of items'''
    pos_imp = sorted(np.random.uniform(low=0.0, high=1.0, size = n), reverse = True)
    item_qual = sorted(np.random.uniform(low=0.0, high=1.0, size= m), reverse = True)    
    if distinct == True:
        properties = gen_distinct_properties(m,p)
    else:
        properties = gen_properties(m,p)

    return pos_imp, item_qual, properties



def get_prop_list_sim(properties):
    '''outputs a sets of items with a given property'''
    prop_list = []
    for prop in range(len(properties[0])):    
        prop_list.append([i for i, j in enumerate(properties.T[prop][:]) if j == 1])
    return prop_list

def get_val_mat(pos_imp, item_qual):
    '''computes the value matrix, w(ij)=u(i)v(j)'''
    m = len(item_qual)
    n = len(pos_imp)
    val_mat= [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            val_mat[i][j] = pos_imp[i] * item_qual[j]
    return val_mat

def topk_upbounds(n, p, prob):
    '''generates random bounds for top-k ranking maximisation'''
    up_bounds = [[0] * (n) for i in range(p)]
    # there is no guarantee of a perfect matching
    for pr in range(p):
        for pos in range(n):
            up_bounds[pr][pos] = up_bounds[pr][pos-1] + int(np.random.binomial(1,prob,1))
    return up_bounds

### YOW news recommendation data ###
# read the dataset
dat_raw = pd.read_excel('yow_userstudy_raw.xls')

# topic data subset
def topic_data(topic, threshold):
    '''  reads the yow data set, chooses a subset of data corresponding to one topic, 
    with a threshold on the constribution of each source of news, 
    returns data on the source (RSS_ID), topic, and relevance'''
    
    dat = pd.read_excel('yow_userstudy_raw.xls')
    dat = dat[['RSS_ID', 'classes', 'relevant']]
    dat_exp = dat.classes.str.split('|', expand=True)
    dat_exp = dat_exp.drop(0, axis=1)
    dat_exp.columns = ['class_1','class_2','class_3','class_4','class_5',
                   'class_6','class_7','class_8','class_9','class_10']
    dat = pd.concat([dat, dat_exp], axis=1)

    dat = dat[(dat.class_1 == topic) | (dat.class_2 == topic) | (dat.class_3 == topic) |
              (dat.class_4 == topic) | (dat.class_5 == topic) | (dat.class_6 == topic) |
              (dat.class_7 == topic) | (dat.class_8 == topic) | (dat.class_9 == topic) |
              (dat.class_10 == topic)]
    dat = dat[['RSS_ID', 'relevant']]
    dat = dat.groupby("RSS_ID").filter(lambda x: len(x) >= threshold)
    dat.index = range(len(dat))
    dat.relevant = dat.relevant.add(np.random.normal(0,0.05, len(dat.relevant)))
    dat = pd.concat([dat.relevant, pd.get_dummies(dat['RSS_ID'])], axis=1)
    dat = dat.sort_values('relevant', ascending=False)
    dat = dat.reset_index(drop=True)

    return dat

# user data subset
def user_data(user):
    '''  reads the yow data set, chooses a subset of data corresponding to one user,
    returns data on the topics, and user likes'''
    dat = pd.read_excel('yow_userstudy_raw.xls')
    dat = dat[['user_id','classes', 'user_like']]
    dat_exp = dat.classes.str.split('|', expand=True)
    dat_exp = dat_exp.drop(0, axis=1)
    dat_exp.columns = ['class_1','class_2','class_3','class_4','class_5',
                   'class_6','class_7','class_8','class_9','class_10']
    dat = pd.concat([dat, dat_exp], axis=1)
    dat = dat[dat.user_id == user]
    dat.fillna(value=np.NaN, inplace=True)
            
    dat_stack = dat[dat_exp.columns].stack()
    dat_stack_counts = dat_stack.value_counts()
    all_classes = list(dat_stack_counts.index)
    dat.index = range(len(dat))
    
    props = np.zeros((len(dat),len(all_classes)), dtype = int)
    dat_mat = dat.iloc[:,3:13].as_matrix()

    for i in range(len(props[:,0])):
        for j in range(len(all_classes)):
            if (dat_mat[i,:] == all_classes[j]).any():
                props[i,j] = 1
    
    props = pd.DataFrame(props, columns = all_classes)
    
    dat.user_like = dat.user_like.add(np.random.normal(0,0.05, len(dat.user_like)))
    
    dat = pd.concat([dat.user_like, props], axis=1)
    dat = dat.sort_values('user_like', ascending=False)
    dat = dat.reset_index(drop=True)

    
    return dat

def exposure_data(n):
    '''returns random eposure of n positions'''
    pos_imp = sorted(np.random.uniform(low=0.0, high=1.0, size=n), reverse=True)
    return pos_imp

def get_prop_list(prop):
    '''outputs a sets of items with a given property'''
    p = prop.shape[1]
    prop = prop.as_matrix()
    prop_list = []
    for i in range(p):    
        prop_list.append([i for i, j in enumerate(prop.T[i][:]) if j == 1])
    return prop_list

def parity_pcrm(prop_list, item_qual, notion, viol_coeff):
    ''' generates demographic parity or disparate treatment bounds for the PCRM model'''
    m = len(item_qual)
    if notion == 'demographic':
        parity = []
        for i in range(len(prop_list)):
            parity.append(len(prop_list[i])/m)
    elif notion == 'utilitarian':
        avg_qual = sum(item_qual)/m
        group_qual = np.zeros(len(prop_list))
        for i in range(len(prop_list)):
            for j in prop_list[i]:
                val = item_qual[j]
                group_qual[i] += val
        for i in range(len(group_qual)):
            group_qual[i] = group_qual[i]/len(prop_list[i])
        parity = []
        for i in range(len(group_qual)):
            parity.append(group_qual[i] / (avg_qual * len(prop_list)))
    parity = np.array(parity) * viol_coeff
            
    return parity

def parity_topk(prop_list, item_qual, n, notion):
    ''' generates demographic parity or disparate treatment bounds for the Top-k RM model'''
    m = len(item_qual)
    p = len(prop_list)
    if notion == 'demographic':
        parity = np.zeros((p,n), dtype = int)
        for l in range(p):
            for k in range(n):
                parity[l][k] = math.ceil(len(prop_list[l])/m * (k+1))
    elif notion == 'utilitarian':
        avg_qual = sum(item_qual)/m
        group_qual = np.zeros(len(prop_list))
        for i in range(len(prop_list)):
            for j in prop_list[i]:
                val = item_qual[j]
                group_qual[i] += val
        for i in range(len(group_qual)):
            group_qual[i] = group_qual[i]/len(prop_list[i])
        parity = np.zeros((p,n), dtype = int)
        for l in range(p):
            for k in range(n):
                parity[l][k] = math.ceil(group_qual[l] / (avg_qual * len(prop_list)) * (k+1))
        
    return parity

### Other ###

def gen_data_lp(item_qual, pos_imp, prop_list, parity):
    '''transforms the data into the format needed for the 
    LP/IP formulation '''
    m,n,p = len(item_qual), len(pos_imp), len(prop_list)
    total_imp = sum(pos_imp)
    
    # value matrix as a list, with [w(0,0), w(0,1), ... , w(1,0), ...], c vector
    val_list = []
    for i in range(m):
        for j in range(n):
            val = item_qual[i] * pos_imp[j]
            ind = 'item' + str(i) + ',' + 'position' + str(j)
            val_list.append([ind, val])
    # constraint bounds (b vector)
    bounds = []
    for i in range(m):
        label = 'item ' + str(i)
        bounds.append([label, 1])
    for j in range(n):
        label = 'position ' + str(j)
        bounds.append([label, 1])
    for l in range(p):
        label = 'parity ' + str(l)
        bounds.append([label, parity[l] * total_imp])    
    # constraint matrix A
    A = [[0] * (n * m) for i in range(m + n + p)]
    for i in range(0, m):
        start = i * n
        A[i][start:(start+n)] = [1] * n
    for j in range(m, n + m):
        start = j - m
        A[j][start::n] = [1] * m
    for l in range(m + n, m + n + p):
        items_w_prop_l = prop_list[l - m - n]
        for k in items_w_prop_l:
            start = k * n
            stop = k * n + n
            A[l][start:stop] = pos_imp

    return val_list, bounds, A














import os
os.chdir('C:/Users/ELITEBOOK840/Desktop/diss/code_public')
### this script analyses the real data from yow newsfeed study as provided in the yow_userstudy_raw.xls

import numpy as np
import pandas as pd
import math
import data_gen as dg
import ranking_algorithms as ra
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
#import importlib
#importlib.reload(ra)

# this is function makes better colourbars
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

# read the data and set the position exposure according to DCG
dat_raw = pd.read_excel('yow_userstudy_raw.xls')
n = 50
#pos_imp = dg.exposure_data(n)
pos_imp = np.zeros(50)
for j in range(n):
    pos_imp[j] = 1/(math.log(j+2))

##### topic data - disjoint properties #####
topic = 'entertainment'
topic_df = dg.topic_data(topic,10)
#topic_df = topic_df.reset_index(drop=True)
item_qual_topic = topic_df.iloc[:,0]
properties_topic = topic_df.iloc[:,1:]
m, p = properties_topic.shape
val_mat = dg.get_val_mat(pos_imp, item_qual_topic)
prop_list_topic = dg.get_prop_list(properties_topic)
parity_pcrm_topic = dg.parity_pcrm(prop_list_topic, item_qual_topic, 'demographic', 1.3)
parity_topk_topic_dem = dg.parity_topk(prop_list_topic, item_qual_topic, n, 'utilitarian')

properties_topic.describe()
prop_prev_user = properties_topic.sum()
prop_util_user = np.matmul(item_qual_topic, properties_topic)
prop_avg_util_user = np.zeros(len(prop_util_user))

for i in range(len(prop_util_user)):
    prop_avg_util_user[i] = prop_util_user[i] / prop_prev_user.iloc[i]

# obtain the unconstrained and fair solutions
# unconstrained
rank_unc = ra.unconstrained_ranking_matching(val_mat)
rank_mat_unc = ra.rank_mat_match(rank_unc[0],m,n)
rank_list_unc = ra.rank_list_from_mat(rank_mat_unc)
unc_val = rank_unc[1]

# pcrm
ip_rank_pcrm = ra.ip_parity(item_qual_topic, pos_imp, prop_list_topic, parity_pcrm_topic)
fair_rank_mat = ra.rank_mat_lp(ip_rank_pcrm[3],m,n)
rank_list_fair_ip = ra.rank_list_from_mat(fair_rank_mat)
fair_ip_val = ip_rank_pcrm[2]

lp_rank_pcrm = ra.lp_parity(item_qual_topic, pos_imp, prop_list_topic, parity_pcrm_topic)
fair_lp_rank_mat = ra.rank_mat_lp(lp_rank_pcrm[3],m,n)
fair_lp_val = lp_rank_pcrm[2]

fair_greedy_pcrm = ra.greedy_parity_two(item_qual_topic, pos_imp, properties_topic, parity_pcrm_topic)    
fair_greedy_pcrm_rank_mat = ra.rank_mat_greedy(fair_greedy_pcrm[0],m,n)
fair_greedy_pcrm_val = fair_greedy_pcrm[1]

# topk
greedy_rank_topk_dem = ra.greedy_topk(properties_topic, parity_topk_topic_dem, pos_imp, item_qual_topic)  
fair_greedy_topk_rank_mat = ra.rank_mat_greedy(greedy_rank_topk_dem[0],m,n)
fair_greedy_topk_val = greedy_rank_topk_dem[1]

# summarise unfairness of the unconstrained ranking and loss of utility in constrained solutions
unfair = ra.max_unfair(rank_list_unc, pos_imp, parity_pcrm_topic, prop_list_topic)
pof_exact =  (unc_val - fair_ip_val)/(unc_val)
pof_approx =  (unc_val - fair_greedy_pcrm_val)/(unc_val)
pof_topk = (unc_val - fair_greedy_topk_val)/(unc_val)
pof_lp =  (unc_val - fair_lp_val)/(unc_val)

# visualise the ranking matrices
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 30,
        }

plt.matshow(rank_mat_unc[0:75,:])
plt.ylabel('item', fontdict=font)
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.text(.5, 8, r'value = ' + str(unc_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_one_unc.pdf')

plt.matshow(fair_rank_mat[0:75,:])
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
#plt.text(.5, 8, r'value = ' + str(fair_ip_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_one_ip.pdf')

plt.matshow(fair_greedy_pcrm_rank_mat[0:75,:])
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
#plt.text(.5, 8, r'value = ' + str(fair_greedy_pcrm_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_one_greedy_pcrm.pdf')

plt.matshow(fair_greedy_topk_rank_mat[0:75,:])
plt.ylabel('item', fontdict=font)
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.text(.5, 8, r'value = ' + str(fair_lp_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_one_greedy_topk.pdf')

im = plt.matshow(fair_lp_rank_mat[0:75,:])
add_colorbar(im)
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
#plt.text(.5, 8, r'value = ' + str(fair_greedy_topk_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_one_lp.pdf')

### user data - non-disjoint properties ###
user_id = 51
user_df = dg.user_data(dat_raw, user_id)
item_qual_user = user_df.iloc[:,0]
properties_user = user_df.iloc[:,1:]
prop_list_user = dg.get_prop_list(properties_user)
m, p = properties_user.shape
val_mat = dg.get_val_mat(pos_imp, item_qual_user)
parity_pcrm_user = dg.parity_pcrm(prop_list_user, item_qual_user, 'demographic', 1.1)
parity_topk_user = dg.parity_topk(prop_list_user, item_qual_user, n, 'demographic')

properties_user.describe()
prop_prev_user = properties_user.sum()
prop_util_user = np.matmul(item_qual_user, properties_user)
prop_avg_util_user = np.zeros(len(prop_util_user))

for i in range(len(prop_util_user)):
    prop_avg_util_user[i] = prop_util_user[i] / prop_prev_user.iloc[i]

# obtain the unconstrained and fair solutions
# unconstrained
rank_unc = ra.unconstrained_ranking_matching(val_mat)
rank_mat_unc = ra.rank_mat_match(rank_unc[0],m,n)
rank_list_unc = ra.rank_list_from_mat(rank_mat_unc)
unc_val = rank_unc[1]

# pcrm
ip_rank_pcrm = ra.ip_parity(item_qual_user, pos_imp, prop_list_user, parity_pcrm_user)
fair_rank_mat = ra.rank_mat_lp(ip_rank_pcrm[3],m,n)
rank_list_fair_ip = ra.rank_list_from_mat(fair_rank_mat)
fair_ip_val = ip_rank_pcrm[2]

lp_rank_pcrm = ra.lp_parity(item_qual_user, pos_imp, prop_list_user, parity_pcrm_user)
fair_lp_rank_mat = ra.rank_mat_lp(lp_rank_pcrm[3],m,n)
fair_lp_val = lp_rank_pcrm[2]

fair_greedy_pcrm = ra.greedy_parity_two(item_qual_user, pos_imp, properties_user, parity_pcrm_user)    
fair_greedy_pcrm_rank_mat = ra.rank_mat_greedy(fair_greedy_pcrm[0],m,n)
fair_greedy_pcrm_val = fair_greedy_pcrm[1]

# summarise unfairness of the unconstrained ranking and loss of utility in constrained solutions
unfair = ra.max_unfair(rank_list_unc, pos_imp, parity_pcrm_user, prop_list_user)
pof_exact =  (unc_val - fair_ip_val)/(unc_val)
pof_approx =  (unc_val - fair_greedy_pcrm_val)/(unc_val)
pof_topk = (unc_val - fair_greedy_topk_val)/(unc_val)
pof_lp =  (unc_val - fair_lp_val)/(unc_val)

# visualise the ranking matrices
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 30,
        }

plt.matshow(rank_mat_unc[0:75,:])
plt.ylabel('item', fontdict=font)
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.text(.5, 8, r'value = ' + str(unc_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_two_unc.pdf')

plt.matshow(fair_rank_mat[0:75,:])
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
#plt.text(.5, 8, r'value = ' + str(fair_ip_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_two_ip.pdf')

plt.matshow(fair_greedy_pcrm_rank_mat[0:75,:])
#add_colorbar(im)
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
#plt.text(.5, 8, r'value = ' + str(fair_greedy_pcrm_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_two_greedy_pcrm.pdf')

im = plt.matshow(fair_lp_rank_mat[0:75,:])
add_colorbar(im)
plt.xlabel('position', fontdict=font)
plt.xticks(fontsize=20)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
#plt.text(.5, 8, r'value = ' + str(fair_greedy_topk_val), fontdict=font)
plt.savefig('C:/Users/ELITEBOOK840/Desktop/diss/text/real_two_lp.pdf')









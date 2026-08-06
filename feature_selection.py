#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:37:36 2026

@author: wynnsu
"""

from feature_extract import (analyze_mount,analyze_attack)
from scipy import stats
import numpy as np
import json
import os 
import matplotlib.pyplot as plt
import time

#redirect
os.chdir('/Users/wynnsu/Downloads/DeepLabCut/notebook_h5_csv')


#open and load the JSON file
with open('calms21_task1_train.json', 'r') as file:
    data_train = json.load(file)
with open('calms21_task1_test.json','r') as file:
    data_test = json.load(file)

'''
Binning features to see if any is redundant

Selected features are implemented to be within DDM-inspired equations

1) Pairwise feature correlations to flag redundancy"
Check whether any two of your 5 candidate features are measuring the same thing. 
Take two features (say, near-hip contact and near-centroid contact) across all your bouts, and compute a correlation coefficient between them (e.g., numpy.corrcoef or scipy.stats.pearsonr). 
Do this for every pair of your 5 features. 
If two features are highly correlated (say, above 0.8), it means knowing one basically tells you the other.
So including both in your final model later would be redundant and could make the fitted model unstable/hard to interpret. 
This step is just: run correlations across all pairs, see which ones are suspiciously high.
'''

#Pair feature to examine correlations and flag redundancy

def corr_analysis(data_train,data_test,vid_id,bout,graph=False):
    
    vid_features = get_lst_for_corr_analysis(data_train,data_test,vid_id,bout)
    
    lst_corr_dist_angle = []
    lst_pval_dist_angle = []
    lst_corr_main_feat = []
    lst_pval_main_feat = []
    
    ###distance vs. angle
    
    for each in range(5):
        
        #do spearman_rank_corr for two features
        corr_dist_angle = spearman_rank_corr(vid_features[0][:,each],vid_features[1][:,each])
        #display graph, show what's returned
        plt.plot(vid_features[0][:,each],vid_features[1][:,each],'o')
        plt.show()
        
        lst_corr_dist_angle.append(corr_dist_angle[0])
        lst_pval_dist_angle.append(corr_dist_angle[1])
        
    ###main features vs. main 
    
    #change array from 3D (features, transition frame, 30) to 2D (features, all frames)
    vid_features_prev = vid_features[3].reshape(vid_features[3].shape[0],-1)
    vid_features_aft  = vid_features[4].reshape(vid_features[4].shape[0],-1)

    #for each feature, compare to rest of other features
    for prev in range(1,5):
        for aft in range(prev,5):
            if prev != 0: #plot bar graphs
                #do spearman_rank_corr for two features
                corr_main_feat = spearman_rank_corr(vid_features_prev[prev,:],vid_features_aft[aft,:])
                #display graph, show what's returned
                plt.plot(vid_features_prev[prev,:],vid_features_aft[aft,:],'o')
                plt.show()
                
                lst_corr_main_feat.append(corr_main_feat[0])
                lst_pval_main_feat.append(corr_main_feat[1])
    
    return np.array([lst_corr_dist_angle,lst_corr_dist_angle],[lst_corr_main_feat,lst_pval_main_feat])

#possible features comparison combos
'''
distance (resident nose & intruder head)     vs. angle_btw_lines(resi_nose,resi_neck,intr_centroid_head)
distance (resident nose & intruder hip)      vs. angle_btw_lines(resi_nose,resi_neck,intr_hipL/R)*
distance (resident nose & intruder centroid) vs. angle_btw_lines(resi_nose,resi_neck,intr_centroid_all)
distance (resident nose & intruder tail)     vs. angle_btw_lines(resi_nose,resi_neck,intr_tail)

["True_Head","True_Hip","True_Centroid","True_Tail","None"] vs. s_matrix/facing_angle/time_since/visual_cone
s_matrix vs. facing_angle/time_since/visual_cone
facing_angle vs. time_since/visual_cone
time_since vs. visual_cone
'''


def spearman_rank_corr(var1,var2):
    
    result = stats.spearmanr(var1,var2)
    
    return (result.statistic,result.pvalue)


def get_lst_for_corr_analysis(data_train,data_test,vid_id,bout):
    
    #analyze mount bouts
    if bout == "mount":
            
        vid_mount = analyze_mount(str(vid_id),data_train,data_test)
            
        #get all pre-bout frames (total = 30 * number of bouts)
        lst_frames_vid = []
        mount_gap = vid_mount.pre_mount()
        for each in mount_gap:
            frames_vid = vid_mount.get_pre_onset(each)
            lst_frames_vid.append(frames_vid)
            
        lst_prev_dist = [] #contains distances during pre-mount frames (X*5; frame, compared bodyparts)
        lst_aft_angle = [] #contains angles during pre-mount frames    (X*5; frame, compared bodyparts)
    
        #loop through transition frame
        for lst in range(len(lst_frames_vid)):
            #loop through pre-transition frames
            for ele in lst_frames_vid[lst]:
                dist =  vid_mount.contact_region_numb(ele,vid_mount.coor_resi,vid_mount.coor_intr)[0,:]
                angle = vid_mount.contact_region_numb(ele,vid_mount.coor_resi,vid_mount.coor_intr)[1,:]
                lst_prev_dist.append(dist)
                lst_aft_angle.append(angle)
                    
        prev_main = vid_mount.stats_arr(False)[:4,:,:]
        aft_main = vid_mount.stats_arr(False)[1:,:,:]
            
        return (np.array(lst_prev_dist),np.array(lst_aft_angle),prev_main,aft_main)
        
    #analyze attack bouts
    else:

        vid_attack = analyze_attack(str(vid_id),data_train,data_test)
            
        #get all pre-bout frames (total = 30 * number of bouts)
        lst_frames_vid = []
        attack_gap = vid_attack.pre_attack()
        for each in attack_gap:
            frames_vid = vid_attack.get_pre_onset(each)
            lst_frames_vid.append(frames_vid)
    
        lst_prev_dist = []
        lst_aft_angle = []

        #loop through transition frame
        for lst in range(len(lst_frames_vid)):
            #loop through pre-transition frames
            for ele in lst_frames_vid[lst]:
                dist =  vid_attack.contact_region_numb(ele,vid_mount.coor_resi,vid_mount.coor_intr)[0,:]
                angle = vid_attack.contact_region_numb(ele,vid_mount.coor_resi,vid_mount.coor_intr)[1,:]
                lst_prev_dist.append(dist)
                lst_aft_angle.append(angle)
                    
        prev_main = vid_attack.stats_arr(False)[:4,:,:]
        aft_main = vid_attack.stats_arr(False)[1:,:,:]
            
        return (np.array(lst_prev_dist),np.array(lst_aft_angle),prev_main,aft_main)


'''
2) Univariate binning per feature for screening + functional form
This is exactly what you've been doing already — one feature at a time, bin it, plot P(escalation) per bin. 
"Univariate" just means "one variable at a time," as opposed to binning on two features simultaneously. 
Nothing new here — just labeling what you already know how to do.
'''


    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:37:36 2026

@author: wynnsu
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import time
from data_loader import data_train, data_test
from feature_extract import (analyze_mount,analyze_attack,analyze_other)

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
    print(vid_features)
    #if the indicated bout does not exist, skip this video
    if vid_features is None:
        return print(f'No {bout} exists in video {vid_id}')
    
    lst_corr_dist_angle = []
    lst_pval_dist_angle = []
    lst_corr_main_feat = []
    lst_pval_main_feat = []
    
    ###distance vs. angle
    
    for each in range(5):
        
        #do spearman_rank_corr for two features
        corr_dist_angle = spearman_rank_corr(vid_features[0][:,each],vid_features[1][:,each])
        lst_corr_dist_angle.append(corr_dist_angle[0])
        lst_pval_dist_angle.append(corr_dist_angle[1])
        
        if graph == True:
            
            #display graph, show what's returned
            plt.plot(vid_features[0][:,each],vid_features[1][:,each],'o')
            plt.title(f'video {vid_id}')
            plt.xlabel("dist")
            plt.ylabel("angle")
            plt.show()
        
    ###main features vs. main 
    
    feature_lst = ["contact_region","s_matrix","facing_angle","time_since","head visual_cone","body visual_cone"]
    
    #change array from 3D (features, transition frame, 30) to 2D (features, transition frames*30)
    vid_features_prev = vid_features[2].reshape(vid_features[2].shape[0],-1)
    vid_features_aft  = vid_features[3].reshape(vid_features[3].shape[0],-1)

    #for each feature, compare to rest of other features
    for prev in range(1,5):
        for aft in range(prev,5):
            #do spearman_rank_corr for two features
            corr_main_feat = spearman_rank_corr(vid_features_prev[prev,:],vid_features_aft[aft,:])
            lst_corr_main_feat.append(corr_main_feat[0])
            lst_pval_main_feat.append(corr_main_feat[1])
            
            if graph == True: #scatter plots
                #display graph, show what's returned
                plt.plot(vid_features_prev[prev,:],vid_features_aft[aft,:],'o')
                plt.title(f'video {vid_id}')
                plt.xlabel(feature_lst[prev])
                plt.ylabel(feature_lst[aft+1])
                plt.show()
                
    return ([lst_corr_dist_angle,lst_pval_dist_angle],[lst_corr_main_feat,lst_pval_main_feat])

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
    
    statistic,pvalue = stats.spearmanr(var1,var2)
    
    return (statistic,pvalue)


def get_lst_for_corr_analysis(data_train,data_test,vid_id,bout):
    
    #analyze mount bouts
    if bout == "mount":
            
        vid_mount = analyze_mount(str(vid_id),data_train,data_test)
            
        # =============================================================================
        #         #get all pre-bout frames (total = 30 * number of bouts)
        #         lst_frames_vid = []
        #         mount_gap = vid_mount.pre_mount()
        #         for each in mount_gap:
        #             frames_vid = vid_mount.get_pre_onset(each)
        #             lst_frames_vid.append(frames_vid)
        # =============================================================================
            
        lst_frames_vid = []  
        mount_gap = vid_mount.filter_bout_seg() #numpy array (XX,2), where element is (beginning frame, end frame)
        
        if np.shape(mount_gap) == (): #edge case - if there is no mount bout in the given video
            return print(f'No mounting bouts occured in video {vid_mount.vid_id}')
        
        #for each frame in gap, store all 30 frames before the frame in a list -> get_pre_onset ()
        for each in mount_gap:
            frames_vid = vid_mount.get_pre_onset(each[0])
            if frames_vid is not None:
                lst_frames_vid.append(frames_vid) 
            
        lst_prev_dist = [] #contains distances during pre-mount frames (X*5; frame, compared bodyparts)
        lst_aft_angle = [] #contains angles during pre-mount frames    (X*5; frame, compared bodyparts)
    
        #loop through transition frame
        for lst in range(len(lst_frames_vid)):
            #loop through pre-transition frames
            for ele in lst_frames_vid[lst]:
                dist =  vid_mount.contact_region_numb(ele,vid_mount.coor_resi,vid_mount.coor_intr)[:,0]
                angle = vid_mount.contact_region_numb(ele,vid_mount.coor_resi,vid_mount.coor_intr)[:,1]
                lst_prev_dist.append(dist)
                lst_aft_angle.append(angle)
                    
        main = vid_mount.stats_arr(False)
        prev_main = main[:-1,:,:]
        aft_main = main[1:,:,:]
            
        return (np.array(lst_prev_dist),np.array(lst_aft_angle),prev_main,aft_main)
        
    #analyze attack bouts
    else:

        vid_attack = analyze_attack(str(vid_id),data_train,data_test)
            
        lst_frames_vid = []  
        attack_gap = vid_attack.filter_bout_seg() #numpy array (XX,2), where element is (beginning frame, end frame)
        
        if np.shape(attack_gap) == (): #edge case - if there is no mount bout in the given video
            return print(f'No mounting bouts occured in video {vid_mount.vid_id}')
        
        #for each frame in gap, store all 30 frames before the frame in a list -> get_pre_onset ()
        for each in attack_gap:
            frames_vid = vid_attack.get_pre_onset(each[0])
            if frames_vid is not None:
                lst_frames_vid.append(frames_vid) 
    
        lst_prev_dist = []
        lst_aft_angle = []

        #loop through transition frame
        for lst in range(len(lst_frames_vid)):
            #loop through pre-transition frames
            for ele in lst_frames_vid[lst]:
                dist =  vid_attack.contact_region_numb(ele,vid_attack.coor_resi,vid_attack.coor_intr)[:,0]
                angle = vid_attack.contact_region_numb(ele,vid_attack.coor_resi,vid_attack.coor_intr)[:,1]
                lst_prev_dist.append(dist)
                lst_aft_angle.append(angle)
                    
        main = vid_attack.stats_arr(False)
        
        prev_main = main[:-1,:,:]
        aft_main = main[1:,:,:]
            
        return (np.array(lst_prev_dist),np.array(lst_aft_angle),prev_main,aft_main)

'''
if __name__ == "__main__": #XXX - KEPT FOR RECORD
    
    
    start = time.time() 
    
    for i in range(2,10):
        print(i)
        x = corr_analysis(data_train, data_test, i, "mount")
        if type(x) == str:
            print(x)
        print(f'Done w/ {i}')
    end = time.time()

    print(f'ALL DONE!!! Took {end-start} sec')
    
    if x is not None:
        print("=====dist-angle======")
        print("correlation coefficient dist-angle")
        print(x[0][0])
        print("pval")
        print(x[0][1])
        print('\n')
        print("======features-features======")
        print("correlation coefficient features-features")
        print(x[1][0])
        print("pval")
        print(x[1][1])
        
    #Conclusion - Spearman's correlational coeffcient shows
    #all proposed feature are either not related or moderately related
    #Thus, all features are kept.
'''


'''
2) Univariate binning per feature for screening + functional form
This is exactly what you've been doing already — one feature at a time, bin it, plot P(escalation) per bin. 
"Univariate" just means "one variable at a time," as opposed to binning on two features simultaneously. 
Nothing new here — just labeling what you already know how to do.
'''

def compute_window(lst):
    '''
    lst is a frame
    Returns a list of frame duration greater than or equal to 15
    '''
    lst_window = []
    #given an lst with length ≥ 15:
    #find beginning and end frame, store as variables
    
    #lst total length//15, store as variables
    loop = len(lst)//15
    #lst total length%15, store as variables
    begin = len(lst)%15
    
    #iterate lst from x to the end
    for i in range(loop):
        store = (begin,begin+14)
        lst_window.append(store)
        begin = begin + 15
    return lst_window
    

def other_s_matrix_calc(class_other,lst_frames):
    '''
    class_other is an instance of analyze_others()
    lst_frames is a list of frame that belongs to non-mount and non-attack frames
    Returns an array of change in s_matrix over 15 frames
    '''
    lst_delt_sm = []
    
    #edge case: if lst_frames is not nested but a list itself
    if isinstance(lst_frames[0], np.int64): 
        windows = compute_window(lst_frames)
        for bout in windows:
            sm_I = class_other.s_matrix(bout[0],class_other.coor_intr,class_other.coor_resi)
            sm_F = class_other.s_matrix(bout[1],class_other.coor_intr,class_other.coor_resi)
            delt_sm = (sm_F-sm_I) / 15
            lst_delt_sm.append(delt_sm)
            
        return lst_delt_sm

    #get a list of 15-frame window, stored as tuples [begin, end of 15 frames]
    for lst in lst_frames:
        #if length of lst is greater than or equal to 15:
        if len(lst) >= 15:
            windows = compute_window(lst)
            for bout in windows:
                sm_I = class_other.s_matrix(bout[0],class_other.coor_intr,class_other.coor_resi)
                sm_F = class_other.s_matrix(bout[1],class_other.coor_intr,class_other.coor_resi)
                delt_sm = (sm_F-sm_I) / 15
                lst_delt_sm.append(delt_sm)
    
    return lst_delt_sm

def m_a_s_matrix_calc(class_ma,lst_frames):
    '''
    class_ma is an instance of either analyze_mount() or analyze_attack()
    lst_frames is a nested list of frame that belongs to mount and attack frames
    Returns an array of change in s_matrix over 15 frames
    '''
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    
    lst_delt_ma = []
    for lst in lst_frames:
        arr_fr = class_ma.get_pre_onset(lst[0])
        if arr_fr is not None:
            for i in [0,15]: #only iterate twice because pre-transition frames are always 30, thus 30/15=2
                ma_I = class_ma.s_matrix(arr_fr[i],class_ma.coor_intr,class_ma.coor_resi)
                ma_F = class_ma.s_matrix(arr_fr[i+14],class_ma.coor_intr,class_ma.coor_resi)
                delt_ma = np.abs((ma_F-ma_I) / 15)
                lst_delt_ma.append(delt_ma)

    return lst_delt_ma

def s_matrix_bin(s_matrix_val,intervals):
    '''
    arr is the array that contains s-matrix throughouth the video
    intervals wished to be implemented (ie how many bins to put)
    
    equal width binning
    
    function inspired by "Binning in Data Mining" in Geeks for Geeks
    '''
    
    if s_matrix_val is None:
        return None
    
    #create an array of bins
    #EXAMPLE: 0.0 ≤ x < 0.1 is 1 bin 
    arr = np.round(np.linspace(0, 0.06, num=intervals+1),10)
    
    bins = []
    
    #for each interval, iterate all elements in s_matrix_val
    for i in range(intervals):
        bin_val = []
        for j in s_matrix_val:
            #if binning the first 9 bins
            if i < intervals-1:
                if arr[i] <= j < arr[i+1]:
                    bin_val.append(j)
            #else binning the last bin
            else:
                if arr[i] <= j <= arr[i+1]:
                    bin_val.append(j)
        bins.append(bin_val)

    return arr,bins

def graph_s_matrix_bin(vid_id,behavior):
    '''
    given all s_matrix values from three diff classes, bin them
    '''
    
    #compute s_matrix for three diff classes
    mount = analyze_mount(str(vid_id),data_train,data_test)
    attack = analyze_attack(str(vid_id),data_train,data_test)
    other = analyze_other(str(vid_id),data_train,data_test)
    
    #get frame duration of behaviors
    dur_mount = mount.filter_bout_seg()
    dur_attack = attack.filter_bout_seg()
    dur_other = other.get_other_frames()
    
    #compute s_matrix rate
    sm_mount = m_a_s_matrix_calc(mount, dur_mount)
    sm_attack = m_a_s_matrix_calc(attack, dur_attack)
    sm_other = other_s_matrix_calc(other,dur_other)
    
    '''
    for i in [sm_mount,sm_attack,sm_other]: #XXX TO BE DELETED
        data = np.array(i)  # your full computed values across all windows
        print("min:", data.min(), "max:", data.max())
    '''
    
    #bin each of them using s_matrix_bin()
    bin_mount = s_matrix_bin(sm_mount, 10)
    bin_attack = s_matrix_bin(sm_attack, 10)
    bin_other = s_matrix_bin(sm_other, 10)
    
    bin_sum = []
    #take the same bin from all classes, compute probability
    #probability of the behavior given a value range
    for i in range(10):
        curr_bin = []
        for j in [bin_mount,bin_attack,bin_other]:
            if j is not None:
                #total number in a specific bin
                curr_bin.extend(j[1][i])
        bin_sum.append(curr_bin)
        
    prob = [] #probability of an indicated given a range of feature value
    #e.g. given s-matrix range [0,0.1], how many of those s-matrix values belong to mount/attack/other
    #show bins given an indicated behavior
    
    plt.figure()
    if behavior == "mount":
        if bin_mount is None:
            return print(f'No mount in video {vid_id}')
        for i in range(10):
            if len(bin_sum[i]) != 0:
                print(f'bin{i}: {len(bin_mount[1][i])},{len(bin_sum[i])}')
                print(f'RESULT: {len(bin_mount[1][i])/len(bin_sum[i])}')
                prob.append(len(bin_mount[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'MOUNT: Video {vid_id}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(MOUNT|Feature)")
        plt.bar(bin_mount[0][:-1],prob,width=np.diff(bin_mount[0]),align='edge')

    elif behavior == "attack":
        if bin_attack is None:
            return print(f'No attack in video {vid_id}')
        for i in range(10):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_attack[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'ATTACK: Video {vid_id}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(MOUNT|Feature)")
        plt.bar(bin_attack[0][:-1],prob,width=np.diff(bin_attack[0]),align='edge')
        
    else: #behavior == "other"
        for i in range(10):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_other[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'OTHER: Video {vid_id}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(MOUNT|Feature)")
        plt.bar(bin_other[0][:-1],prob,width=np.diff(bin_other[0]),align='edge')
    plt.show()
    
    return prob

#probability of the behavior given a value range
if __name__ == "__main__":
    
    start = time.time()
    
    #set up
    vid1_mount = analyze_mount("1", data_train, data_test)
    vid1_attack = analyze_attack("1", data_train, data_test)
    vid1_other = analyze_other("1", data_train, data_test)
    
    '''
    vid1_mount_stats = vid1_mount.stats_arr(False)
    vid1_attack_stats = vid1_attack.stats_arr(False) #None
    vid1_other_stats = vid1_other.stats_arr(False)
    
    lst = [vid1_mount,vid1_attack,vid1_other]
        
    vid1_sm = [np.reshape(vid1_mount_stats[1,:,:],-1),vid1_other_stats[1,:]]
    print(np.shape(vid1_sm[0]),np.shape(vid1_sm[1]))
    
    x = s_matrix_bin(vid1_sm[0], 10)
    assert np.shape(vid1_sm[0])[0] == len(np.concatenate(x[1]))
    '''

    
    for i in range (11,13):
        
        graph_s_matrix_bin(i, "mount")
        graph_s_matrix_bin(i, "attack")
        graph_s_matrix_bin(i, "other")
     
    
    end = time.time()
    
    print(f'DONE!!! Took {end-start} seconds')

'''
Step 3: fitting drift-rate equations and comparing them
This is the actual model-fitting stage, not binning anymore. You write down a few candidate formulas for how drift rate depends on your features — for example:
Equation A (additive, no interactions): drift = b0 + b1*hip_contact + b2*sym_ratio_change + b3*time_since_last_bout + ...
Equation B (with an interaction term): drift = b0 + b1*hip_contact + b2*sym_ratio_change + b3*(hip_contact * sym_ratio_change) + ...
"Additive" means each feature just adds its own independent contribution. 
"Interaction term" (the hip_contact * sym_ratio_change part) means the effect of one feature depends on the value of another — this is how you'd formally represent whatever you found in step 3's 2D grid, if anything looked interesting there.
You then fit each candidate equation to your data (find the best-fitting b0, b1, b2... values) and need a way to decide which equation is actually better, not just which one fits your existing data best (a model with more terms will always fit existing data at least as well, even if the extra terms are noise).
'''

    
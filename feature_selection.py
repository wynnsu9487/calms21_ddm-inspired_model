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
import sys
print(sys.executable)
import statsmodels.api as sm


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
    '''
    Returns lists of features to be compared for correlational analysis
    
    Calls the function get_lst_for_corr_analysis(), extracts desired features, 
    stores desired features at specific frames in lists, graph desired features
    to compare them by Spearman correlational coefficient
    
    Parameters:
        data_train: (dict) xy coordinates from both mice, from all videos for training
        data_test:  (dict) xy coordinates from both mice, from all videos for testing
        vid_id:     (int) video number/id
        bout:       (str) behavior to be analyzed & extracted
        graph:      (bool) graphs pairwise features 
    '''
    vid_features = get_lst_for_corr_analysis(data_train,data_test,vid_id,bout)
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

def spearman_rank_corr(var1,var2):
    '''
    Returns correlation coefficient & its p-value given two features
    
    Parameter:
        var1: (list) a feature analyzed with var2
        var2: (list) a feature analyzed with var1
    '''
    
    statistic,pvalue = stats.spearmanr(var1,var2)
    
    return (statistic,pvalue)


def get_lst_for_corr_analysis(data_train,data_test,vid_id,bout):
    '''
    Returns a list of arrays that contain features to be compared:
        distance (resident nose & intruder head)     vs. angle_btw_lines(resi_nose,resi_neck,intr_centroid_head)
        distance (resident nose & intruder hip)      vs. angle_btw_lines(resi_nose,resi_neck,intr_hipL/R)*
        distance (resident nose & intruder centroid) vs. angle_btw_lines(resi_nose,resi_neck,intr_centroid_all)
        distance (resident nose & intruder tail)     vs. angle_btw_lines(resi_nose,resi_neck,intr_tail)

        ["True_Head","True_Hip","True_Centroid","True_Tail","None"] vs. s_matrix/facing_angle/time_since/visual_cone
        s_matrix vs. facing_angle/time_since/visual_cone
        facing_angle vs. time_since/visual_cone
        time_since vs. visual_cone
        
    Used by corr_analysis to provide correlation coefficients of features & p-value of the coefficients
    
    Parameter:
        data_train: (dict) xy coordinates from both mice, from all videos for training
        data_test:  (dict) xy coordinates from both mice, from all videos for testing
        vid_id:     (str) video number/id
        bout:       (str) behavior to be analyzed & extracted
    '''
    #analyze mount bouts
    if bout == "mount":
            
        vid_mount = analyze_mount(str(vid_id),data_train,data_test)
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

# =============================================================================
# Contact Region Binning
# =============================================================================

#P({MOUNT,ATTACK,OTHER} | {True_Head,True_Centroid,True_Hip,True_Tail,None})

def count_match(match,arr):
    '''
    Returns how many elements in arr matches the string in match
    
    Used by cr_xxx(), or functions contact_region() binning 
    
    Parameter:
        match: (str) a word that wished to be matched
        arr:   (list) a list to be iterated to find how many elements match the word
    '''
    count = 0
    for i in range(len(arr)):
            if arr[i] == match:
                count = count + 1
                
    return count

#get contact_region for specific behavior:
def cr_mount(class_mount,lst_frames):
    '''
    Returns bins of contact regions {True_Head,True_Hip,True_Centroid,True_Tail,None},
    given that the behavior is mount.
    
    ie how many contact_regions occured before mount happened
    
    Parameters:
        class_mount: (class) an instance of analyze_mount
        lst_frames:  (lst) list of durations when mount happened
    '''
    regions = ["True_Head","True_Hip","True_Centroid","True_Tail","None"]
    
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    lst_contact = [] #stores a list of regions being contacted
    for lst in lst_frames:
        arr_fr = class_mount.get_pre_onset(lst[0])
        if arr_fr is not None:
            for fr in arr_fr:
                contacted = class_mount.contact_region(fr,class_mount.coor_resi,class_mount.coor_intr)
                lst_contact.append(contacted) #append the contacted region
    
    bins = [] #organized by {"True_Head","True_Hip","True_Centroid","True_Tail","None"}
    for each in regions:
        bins.append(count_match(each,lst_contact)) 
        
    return bins
    
def cr_attack(class_attack,lst_frames):
    '''
    Returns bins of contact regions {True_Head,True_Hip,True_Centroid,True_Tail,None},
    given that the behavior is attack.
    
    ie how many contact_regions occured before attack happened
    
    Parameters:
        class_mount: (class) an instance of analyze_attack
        lst_frames:  (lst) list of durations when attack happened
    
    '''
    regions = ["True_Head","True_Hip","True_Centroid","True_Tail","None"]
    
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    
    bins = [] #organized by {True_Head,True_Centroid,True_Hip,True_Tail,None}
    lst_contact = [] #stores a list of regions being contacted
    for lst in lst_frames:
        arr_fr = class_attack.get_pre_onset(lst[0])
        if arr_fr is not None:
            for fr in arr_fr:
                contacted = class_attack.contact_region(fr,class_attack.coor_resi,class_attack.coor_intr)
                lst_contact.append(contacted) #append the contacted region
    
    for each in regions:
        bins.append(count_match(each,lst_contact))
    return bins

def cr_other(class_other,lst_frames):
    '''
    Returns bins of contact regions {True_Head,True_Hip,True_Centroid,True_Tail,None},
    given that the behavior is other.
    
    ie how many contact_regions occured before other behaviors happened
    
    Parameters:
        class_mount: (class) an instance of analyze_other
        lst_frames:  (lst) list of durations when other behaviors happened
    '''
    
    regions = ["True_Head","True_Hip","True_Centroid","True_Tail","None"]
    
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    bins = [] #organized by {True_Head,True_Centroid,True_Hip,True_Tail,None}
    lst_contact = [] #stores a list of regions being contacted
    for lst in lst_frames:
        if not isinstance(lst, np.int64): 
            for fr in lst:
                contacted = class_other.contact_region(fr,class_other.coor_resi,class_other.coor_intr)
                lst_contact.append(contacted) #append the contacted region
        else:
            contacted = class_other.contact_region(lst,class_other.coor_resi,class_other.coor_intr)
            lst_contact.append(contacted)
    
    for each in regions:
        bins.append(count_match(each,lst_contact))
        
    return bins

def graph_cr_bin(vid_id,behavior):
    '''
    Returns bins of probability of contact regions {True_Head,True_Hip,True_Centroid,True_Tail,None}
    for a video, given a behavior
    
    ie P(mount/attack/other | regions): probaility of a behavior given a region is contacted
    
    Parameters:
        vid_id: (int) video number/id
        behavior: (str) behavior hopes to be binned
    '''
    
    #get instance of mount, attack, other from a video
    mount = analyze_mount(str(vid_id),data_train,data_test)
    attack = analyze_attack(str(vid_id),data_train,data_test)
    other = analyze_other(str(vid_id),data_train,data_test)
    
    #get bout duration for indicated behaviors
    dur_mount = mount.filter_bout_seg()
    dur_attack = attack.filter_bout_seg()
    dur_other = other.get_other_frames()
    
    #get frequency of contacts for indicated behaviors, 
    #where each bin represents a contact region
    bins_mount = cr_mount(mount,dur_mount)
    bins_attack = cr_attack(attack,dur_attack)
    bins_other = cr_other(other,dur_other)
    
    #get total of bin_mount/attack/other
    bin_sum = []
    
    #take the same bin from all classes, compute probability
    #probability of the behavior given a value range
    for i in range(5):
        total = 0
        for j in [bins_mount,bins_attack,bins_other]:
            if j is not None:
                #total number in a specific bin
                total = j[i] + total
        bin_sum.append(total)
    
    prob = []
    plt.figure()
    #bin is each contact region made before bout / total contacts made
    if behavior == "mount":
        if bins_mount is None:
            return print(f'No mount in video {vid_id}')
        for i in range(5):
            if bin_sum[i] != 0:
                prob.append(bins_mount[i]/bin_sum[i])
            else:
                prob.append(0)
        plt.title(f'MOUNT: Video {vid_id}')
        plt.xlabel("Contact Region")
        plt.ylabel("P(MOUNT | Feature)")
        plt.bar(["True_Head","True_Hip","True_Centroid","True_Tail","None"],prob,align='edge')
        
    elif behavior == "attack":
        if bins_attack is None:
            return print(f'No attack in video {vid_id}')
        for i in range(5):
            if bin_sum[i] != 0:
                prob.append(bins_attack[i]/bin_sum[i])
            else:
                prob.append(0)
        plt.title(f'ATTACK: Video {vid_id}')
        plt.xlabel("Contact Region")
        plt.ylabel("P(ATTACK | Feature)")
        plt.bar(["True_Head","True_Hip","True_Centroid","True_Tail","None"],prob,align='edge')
                
    else: #behavior == "other"
        if bins_other is None:
            return print(f'No other in video {vid_id}')
        for i in range(5):
            if bin_sum[i] != 0:
                prob.append(bins_other[i]/bin_sum[i])
            else:
                prob.append(0)
        plt.title(f'OTHER: Video {vid_id}')
        plt.xlabel("Contact Region")
        plt.ylabel("P(OTHER | Feature)")
        plt.bar(["True_Head","True_Hip","True_Centroid","True_Tail","None"],prob,align='edge')
            
    plt.show()
    return prob

# =============================================================================
# S-Matrix Binning
# =============================================================================

def compute_window(lst):
    '''
    Returns a list of windows (duration) greater than or equal to 15 frames
    
    Parameter:
        lst: (list) frames that are pre-bout transition frames (or other frames)
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
    Returns an array of change in s_matrix over 15 frames, given that other behaviors happened
    
    Parameters:
        class_other: (class) an instance of analyze_other()
        lst_frames: (list) a nested list of other frames; each list in lst_frames is
        a sustained period of other behaviors.
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
    Returns an array of change in s_matrix over 15 frames, given that mount or attack behaviors happened
    
    Parameters:
        class_other: (class) an instance of analyze_other()
        lst_frames: (list) a nested list of mount or attack frames; each list in lst_frames 
        is a sustained period of mount or attack behaviors.
    '''
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None

    lst_delt_ma = []
    for lst in lst_frames:
        arr_fr = class_ma.get_pre_onset(lst[0])
        if arr_fr is not None: #edge case: if beginning of pre-transition is < 0
            for i in [0,15]: #only iterate twice because pre-transition frames are always 30, thus 30/15=2
                ma_I = class_ma.s_matrix(arr_fr[i],class_ma.coor_intr,class_ma.coor_resi)
                ma_F = class_ma.s_matrix(arr_fr[i+14],class_ma.coor_intr,class_ma.coor_resi)
                delt_ma = (ma_F-ma_I) / 15
                lst_delt_ma.append(delt_ma)

    return lst_delt_ma

def s_matrix_bin(s_matrix_val,intervals):
    '''
    Returns an array of feature vectors & bins containing actual features
    
    Parameters:
        s_matrix_val: (array) elements are change in s_matrix value over 15 frames
        intervals: (int) indicates how many bins to be divided
    
    #NOTE: this is an equal width binning function 
    inspired by "Binning in Data Mining" in Geeks for Geeks
    '''
    
    if s_matrix_val is None:
        return None
    
    #create an array of bins
    #EXAMPLE: 0.0 ≤ x < 0.1 is 1 bin 
    arr = np.round(np.linspace(-0.05, 0.06, num=intervals+1),10)
    
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

def graph_s_matrix_bin(vid_id,behavior,intervals=20):
    '''
    Returns an array of probailities of an action being a behavior
    given feature vectors being binned
    
    Parameters:
        vid_id: (int) video number/id
        behavior: (str) a behavior {mount,attack,other} to be binned
        intervals: (int) number of bins to be used
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
    bin_mount = s_matrix_bin(sm_mount, intervals)
    bin_attack = s_matrix_bin(sm_attack, intervals)
    bin_other = s_matrix_bin(sm_other, intervals)
    
    bin_sum = []
    #take the same bin from all classes, compute probability
    #probability of the behavior given a value range
    for i in range(intervals):
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
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_mount[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'MOUNT: Video {vid_id}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(MOUNT | Feature)")
        plt.bar(bin_mount[0][:-1],prob,width=np.diff(bin_mount[0]),align='edge') #equal width

    elif behavior == "attack":
        if bin_attack is None:
            return print(f'No attack in video {vid_id}')
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_attack[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'ATTACK: Video {vid_id}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(ATTACK | Feature)")
        plt.bar(bin_attack[0][:-1],prob,width=np.diff(bin_attack[0]),align='edge')
        
    else: #behavior == "other"
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_other[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'OTHER: Video {vid_id}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(OTHER | Feature)")
        plt.bar(bin_other[0][:-1],prob,width=np.diff(bin_other[0]),align='edge')
    plt.show()
    
    return prob

# =============================================================================
# Facing Angle Binning
# =============================================================================

#NOTE - still use compute_window() from s_matrix

def fa_mount(class_mount,lst_frames):
    '''
    Returns an list of change in resident's facing angle to intruders 
    over 15 frames for mount behaviors
    
    Parameters:
        class_mount: (class) an instance of analyze_mount()
        lst_frames: (list) a nested list containing all mount bouts
    '''
    
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    
    lst_delt_fa = []
    for lst in lst_frames:
        arr_fr = class_mount.get_pre_onset(lst[0])
        if arr_fr is not None: #edge case: if beginning of pre-transition is < 0
            for i in [0,15]: #only iterate twice because pre-transition frames are always 30, thus 30/15=2
                fa_I = class_mount.facing_angle(arr_fr[i],class_mount.coor_intr,class_mount.coor_resi)
                fa_F = class_mount.facing_angle(arr_fr[i+14],class_mount.coor_intr,class_mount.coor_resi)
                delt_fa = np.abs((fa_F-fa_I) / 15)
                lst_delt_fa.append(delt_fa)

    return lst_delt_fa

def fa_attack(class_attack,lst_frames):
    '''
    Returns an list of change in resident's facing angle to intruders 
    over 15 frames for attack behaviors
    
    Parameters:
        class_mount: (class) an instance of analyze_attack()
        lst_frames: (list) a nested list containing all attack bouts
    '''
    
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    
    lst_delt_fa = []
    for lst in lst_frames:
        arr_fr = class_attack.get_pre_onset(lst[0])
        if arr_fr is not None: #edge case: if beginning of pre-transition is < 0
            for i in [0,15]: #only iterate twice because pre-transition frames are always 30, thus 30/15=2
                fa_I = class_attack.facing_angle(arr_fr[i],class_attack.coor_intr,class_attack.coor_resi)
                fa_F = class_attack.facing_angle(arr_fr[i+14],class_attack.coor_intr,class_attack.coor_resi)
                delt_fa = np.abs((fa_F-fa_I) / 15)
                lst_delt_fa.append(delt_fa)

    return lst_delt_fa

def fa_other(class_other,lst_frames):
    '''
    Returns an list of change in resident's facing angle to intruders 
    over 15 frames for other behaviors
    
    Parameters:
        class_mount: (class) an instance of analyze_other()
        lst_frames: (list) a nested list containing all other bouts
    '''
    
    lst_delt_fa = []
    
    #edge case: if lst_frames is not nested but a list itself
    if isinstance(lst_frames[0], np.int64): 
        windows = compute_window(lst_frames)
        for bout in windows:
            fa_I = class_other.facing_angle(bout[0],class_other.coor_intr,class_other.coor_resi)
            fa_F = class_other.facing_angle(bout[1],class_other.coor_intr,class_other.coor_resi)
            delt_fa = (fa_F-fa_I) / 15
            lst_delt_fa.append(delt_fa)
            
        return lst_delt_fa

    #get a list of 15-frame window, stored as tuples [begin, end of 15 frames]
    for lst in lst_frames:
        #if length of lst is greater than or equal to 15:
        if len(lst) >= 15:
            windows = compute_window(lst)
            for bout in windows:
                fa_I = class_other.facing_angle(bout[0],class_other.coor_intr,class_other.coor_resi)
                fa_F = class_other.facing_angle(bout[1],class_other.coor_intr,class_other.coor_resi)
                delt_fa = (fa_F-fa_I) / 15
                lst_delt_fa.append(delt_fa)
    
    return lst_delt_fa

def fa_bins(fa_val,intervals):
    '''
    Returns an array of feature vectors & bins containing actual features
    
    Parameters:
        fa_val: (array) elements are change in facing_angle value over 15 frames
        intervals: (int) indicates how many bins to be divided
    
    #NOTE: this is an equal width binning function 
    inspired by "Binning in Data Mining" in Geeks for Geeks
    '''
    
    if fa_val is None:
        return None
    
    #create an array of bins
    #EXAMPLE: 0.0 ≤ x < 0.1 is 1 bin 
    arr = np.round(np.linspace(0, 20, num=intervals+1),10)
    
    bins = []
    
    #for each interval, iterate all elements in s_matrix_val
    for i in range(intervals):
        bin_val = []
        for j in fa_val:
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

def graph_fa_bins(vid_id,behavior):
    '''
    Returns an array of probailities of an action being a behavior
    given feature vectors being binned
    
    Parameters:
        vid_id: (int) video number/id
        behavior: (str) a behavior {mount,attack,other} to be binned
        intervals: (int) number of bins to be used
    '''
    
    #compute s_matrix for three diff classes
    mount = analyze_mount(str(vid_id),data_train,data_test)
    attack = analyze_attack(str(vid_id),data_train,data_test)
    other = analyze_other(str(vid_id),data_train,data_test)
    
    #get frame duration of behaviors
    dur_mount = mount.filter_bout_seg()
    dur_attack = attack.filter_bout_seg()
    dur_other = other.get_other_frames()
    
    #compute facing_angle rate
    ma_mount = fa_mount(mount, dur_mount)
    ma_attack = fa_attack(attack, dur_attack)
    ma_other = fa_other(other,dur_other)
    
    '''
    for i in [ma_mount,ma_attack,ma_other]: #XXX TO BE DELETED
        data = np.array(i)  # your full computed values across all windows
        print("min:", data.min(), "max:", data.max())
    '''
    
    #bin each of them using s_matrix_bin()
    bin_mount = fa_bins(ma_mount, 10)
    bin_attack = fa_bins(ma_attack, 10)
    bin_other = fa_bins(ma_other, 10)
    
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
                prob.append(len(bin_mount[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'MOUNT: Video {vid_id}')
        plt.xlabel("change in facing_angle over 15 frames")
        plt.ylabel("P(MOUNT | Feature)")
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
        plt.xlabel("change in facing_angle over 15 frames")
        plt.ylabel("P(ATTACK | Feature)")
        plt.bar(bin_attack[0][:-1],prob,width=np.diff(bin_attack[0]),align='edge')
        
    else: #behavior == "other"
        for i in range(10):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_other[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'OTHER: Video {vid_id}')
        plt.xlabel("change in facing_angle over 15 frames")
        plt.ylabel("P(OTHER | Feature)")
        plt.bar(bin_other[0][:-1],prob,width=np.diff(bin_other[0]),align='edge')
    plt.show()
    
    return prob

# =============================================================================
# Visual Cone Binning
# =============================================================================

#NOTE - still use compute_window() from s_matrix

def vc_mount(class_mount,lst_frames,part):
    '''
    Returns an list of change in resident's visual cone to intruders 
    over 15 frames for mount behaviors
    
    Parameters:
        class_mount: (class) an instance of analyze_mount()
        lst_frames: (list) a nested list containing all mount bouts
    '''
    
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    
    lst_delt_vc = []
    for lst in lst_frames:
        arr_fr = class_mount.get_pre_onset(lst[0])
        if arr_fr is not None: #edge case: if beginning of pre-transition is < 0
            for i in [0,15]: #only iterate twice because pre-transition frames are always 30, thus 30/15=2
                fa_I = np.array(class_mount.visual_cone(arr_fr[i],class_mount.coor_intr,class_mount.coor_resi))
                fa_F = np.array(class_mount.visual_cone(arr_fr[i+14],class_mount.coor_intr,class_mount.coor_resi))
                delt_vc = np.abs((fa_F-fa_I) / 15)
                lst_delt_vc.append(delt_vc)
            
    #returns one of the arrays: either head or body
    if part == "head":
        return np.array(lst_delt_vc)[:,0]
    else: #body
        return np.array(lst_delt_vc)[:,1]

def vc_attack(class_attack,lst_frames,part):
    '''
    Returns an list of change in resident's visual cone to intruders 
    over 15 frames for attack behaviors
    
    Parameters:
        class_mount: (class) an instance of analyze_attack()
        lst_frames: (list) a nested list containing all attack bouts
    '''
    
    if lst_frames is None: #edge case: if a specific behavior wasn't in the video
        return None
    
    lst_delt_vc = []
    for lst in lst_frames:
        arr_fr = class_attack.get_pre_onset(lst[0])
        if arr_fr is not None: #edge case: if beginning of pre-transition is < 0
            for i in [0,15]: #only iterate twice because pre-transition frames are always 30, thus 30/15=2
                fa_I = np.array(class_attack.visual_cone(arr_fr[i],class_attack.coor_intr,class_attack.coor_resi))
                fa_F = np.array(class_attack.visual_cone(arr_fr[i+14],class_attack.coor_intr,class_attack.coor_resi))
                delt_vc = np.abs((fa_F-fa_I) / 15)
                lst_delt_vc.append(delt_vc)
                
    #returns one of the arrays: either head or body
    if part == "head":
        return np.array(lst_delt_vc)[:,0]
    else: #body
        return np.array(lst_delt_vc)[:,1]

def vc_other(class_other,lst_frames,part):
    '''
    Returns an list of change in resident's visual cone to intruders 
    over 15 frames for other behaviors
    
    Parameters:
        class_mount: (class) an instance of analyze_other()
        lst_frames: (list) a nested list containing all other bouts
    '''
    
    lst_delt_vc = []
    
    #edge case: if lst_frames is not nested but a list itself
    if isinstance(lst_frames[0], np.int64): 
        windows = compute_window(lst_frames)
        for bout in windows:
            fa_I = np.array(class_other.visual_cone(bout[0],class_other.coor_intr,class_other.coor_resi))
            fa_F = np.array(class_other.visual_cone(bout[1],class_other.coor_intr,class_other.coor_resi))
            delt_vc = (fa_F-fa_I) / 15
            lst_delt_vc.append(delt_vc)
            
        #returns one of the arrays: either head or body
        if part == "head":
            return np.array(lst_delt_vc)[:,0]
        else: #body
            return np.array(lst_delt_vc)[:,1]

    #get a list of 15-frame window, stored as tuples [begin, end of 15 frames]
    for lst in lst_frames:
        #if length of lst is greater than or equal to 15:
        if len(lst) >= 15:
            windows = compute_window(lst)
            for bout in windows:
                fa_I = np.array(class_other.visual_cone(bout[0],class_other.coor_intr,class_other.coor_resi))
                fa_F = np.array(class_other.visual_cone(bout[1],class_other.coor_intr,class_other.coor_resi))
                delt_vc = (fa_F-fa_I) / 15
                lst_delt_vc.append(delt_vc)
    
    #returns one of the arrays: either head or body
    if part == "head":
        return np.array(lst_delt_vc)[:,0]
    else: #body
        return np.array(lst_delt_vc)[:,1]

def vc_bins(vc_val,intervals):
    '''
    Returns an array of feature vectors & bins containing actual features
    
    Parameters:
        fa_val: (array) elements are change in visual_cone value over 15 frames
        intervals: (int) indicates how many bins to be divided
    
    #NOTE: this is an equal width binning function 
    inspired by "Binning in Data Mining" in Geeks for Geeks
    '''
    
    if vc_val is None:
        return None
    
    #create an array of bins
    #EXAMPLE: 0.0 ≤ x < 0.1 is 1 bin 
    arr = np.round(np.linspace(-0.30, 0.30, num=intervals+1),10)
    
    bins = []
    
    #for each interval, iterate all elements in s_matrix_val
    for i in range(intervals):
        bin_val = []
        for j in vc_val:
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

def graph_vc_bins(vid_id,behavior,part):
    '''
    Returns an array of probailities of an action being a behavior
    given feature vectors being binned
    
    Parameters:
        vid_id: (int) video number/id
        behavior: (str) a behavior {mount,attack,other} to be binned
        intervals: (int) number of bins to be used
    '''
    
    #compute s_matrix for three diff classes
    mount = analyze_mount(str(vid_id),data_train,data_test)
    attack = analyze_attack(str(vid_id),data_train,data_test)
    other = analyze_other(str(vid_id),data_train,data_test)
    
    #get frame duration of behaviors
    dur_mount = mount.filter_bout_seg()
    dur_attack = attack.filter_bout_seg()
    dur_other = other.get_other_frames()
    
    
    #compute facing_angle rate
    ma_mount = vc_mount(mount, dur_mount,part)
    ma_attack = vc_attack(attack, dur_attack,part)
    ma_other = vc_other(other,dur_other,part)
    
    '''
    for i in [sm_mount,sm_attack,sm_other]: #XXX TO BE DELETED
        data = np.array(i)  # your full computed values across all windows
        print("min:", data.min(), "max:", data.max())
    '''
    
    #bin each of them using s_matrix_bin()
    bin_mount = vc_bins(ma_mount, 10)
    bin_attack = vc_bins(ma_attack, 10)
    bin_other = vc_bins(ma_other, 10)
    
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
    #e.g. given s-matrix range [0,0.1], how many of those values belong to mount/attack/other
    #show bins given an indicated behavior
    
    plt.figure()
    if behavior == "mount":
        if bin_mount is None:
            return print(f'No mount in video {vid_id}')
        for i in range(10):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_mount[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'MOUNT: Video {vid_id}, {part}')
        plt.xlabel("change in visual_cone over 15 frames")
        plt.ylabel("P(MOUNT | Feature)")
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
        plt.title(f'ATTACK: Video {vid_id}, {part}')
        plt.xlabel("change in visual_cone over 15 frames")
        plt.ylabel("P(ATTACK | Feature)")
        plt.bar(bin_attack[0][:-1],prob,width=np.diff(bin_attack[0]),align='edge')
        
    else: #behavior == "other"
        for i in range(10):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_other[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'OTHER: Video {vid_id}, {part}')
        plt.xlabel("change in visual_cone over 15 frames")
        plt.ylabel("P(OTHER | Feature)")
        plt.bar(bin_other[0][:-1],prob,width=np.diff(bin_other[0]),align='edge')
    plt.show()
    
    return prob

# =============================================================================
# Combining bins across videos
# =============================================================================

def cr_graph_combine(vid_id_range,behavior,intervals=20):
    '''
    Binning for contact_region()
    '''
    
    lst_bins_mount = np.zeros(5)
    lst_bins_attack = np.zeros(5)
    lst_bins_other = np.zeros(5)
    
    #do this for each video
    for i in range (vid_id_range[0],vid_id_range[1]):
        #find each behavior's bins
        #compute s_matrix for three diff classes
        mount = analyze_mount(str(i),data_train,data_test)
        attack = analyze_attack(str(i),data_train,data_test)
        other = analyze_other(str(i),data_train,data_test)
        
        #get frame duration of behaviors
        dur_mount = mount.filter_bout_seg()
        dur_attack = attack.filter_bout_seg()
        dur_other = other.get_other_frames()
        
        #bin behaviors by {True_Head,True_Centroid,True_Hip,True_Tail,None}
        bins_mount = cr_mount(mount,dur_mount)
        bins_attack = cr_attack(attack,dur_attack)
        print(bins_attack)
        bins_other = cr_other(other,dur_other)
        
        #put them in separate lists
        if bins_mount is not None:  
            lst_bins_mount = np.array(bins_mount) + lst_bins_mount
        if bins_attack is not None:
            lst_bins_attack = np.array(bins_attack) + lst_bins_attack
        if bins_other is not None:
            lst_bins_other = np.array(bins_other) + lst_bins_other
    
    #take the same bin from all classes, compute probability
    #probability of the behavior given a value range
    bin_sum = lst_bins_mount + lst_bins_attack + lst_bins_other
    
    prob = [] #probability of an indicated given a range of feature value
    #e.g. given s-matrix range [0,0.1], how many of those s-matrix values belong to mount/attack/other
    #show bins given an indicated behavior
    
    plt.figure()
    if behavior == "mount":
        if bins_mount is None:
            return print(f'No mount in videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        for i in range(5):
            if bin_sum[i] != 0:
                prob.append(lst_bins_mount[i]/bin_sum[i])
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'MOUNT: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(MOUNT | Feature)")
        plt.bar(["True_Head","True_Hip","True_Centroid","True_Tail","None"],prob,align='edge') #equal width

    elif behavior == "attack":
        if bins_attack is None:
            return print(f'No attack in videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        for i in range(5):
            if bin_sum[i] != 0:
                prob.append(lst_bins_attack[i]/bin_sum[i])
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'ATTACK: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(ATTACK | Feature)")
        plt.bar(["True_Head","True_Hip","True_Centroid","True_Tail","None"],prob,align='edge') #equal width
        print(lst_bins_attack)
        print(bin_sum)
    else: #behavior == "other"
        for i in range(5):
            if bin_sum[i] != 0:
                prob.append(lst_bins_other[i]/bin_sum[i])
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'OTHER: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(OTHER | Feature)")
        plt.bar(["True_Head","True_Hip","True_Centroid","True_Tail","None"],prob,align='edge') #equal width
    plt.show()
    
    return bin_sum

def sm_graph_combine(vid_id_range,behavior,intervals=20):
    '''
    Binning for s_matrix
    '''
    
    lst_bins_mount = []
    lst_bins_attack = []
    lst_bins_other = []
    
    #do this for each video
    for i in range (vid_id_range[0],vid_id_range[1]):
        #find each behavior's bins
        #compute s_matrix for three diff classes
        mount = analyze_mount(str(i),data_train,data_test)
        attack = analyze_attack(str(i),data_train,data_test)
        other = analyze_other(str(i),data_train,data_test)
        
        #get frame duration of behaviors
        dur_mount = mount.filter_bout_seg()
        dur_attack = attack.filter_bout_seg()
        dur_other = other.get_other_frames()
        
        #compute s_matrix rate
        ma_mount = m_a_s_matrix_calc(mount, dur_mount)
        ma_attack = m_a_s_matrix_calc(attack, dur_attack)
        ma_other = other_s_matrix_calc(other,dur_other)
        
        #put them in separate lists
        lst_bins_mount.extend(ma_mount if ma_mount is not None else [])
        lst_bins_attack.extend(ma_attack if ma_attack is not None else[])
        lst_bins_other.extend(ma_other if ma_other is not None else [])

    #binning features from given videos
    bin_mount = s_matrix_bin(lst_bins_mount, intervals)
    bin_attack = s_matrix_bin(lst_bins_attack, intervals)
    bin_other = s_matrix_bin(lst_bins_other, intervals)
    
    bin_sum = []
    #take the same bin from all classes, compute probability
    #probability of the behavior given a value range
    for i in range(intervals):
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
            return print(f'No mount in videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_mount[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'MOUNT: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(MOUNT | Feature)")
        plt.bar(bin_mount[0][:-1],prob,width=np.diff(bin_mount[0]),align='edge') #equal width

    elif behavior == "attack":
        if bin_attack is None:
            return print(f'No attack in videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_attack[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'ATTACK: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(ATTACK | Feature)")
        plt.bar(bin_attack[0][:-1],prob,width=np.diff(bin_attack[0]),align='edge')
        
    else: #behavior == "other"
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_other[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'OTHER: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in sym_ratio over 15 frames")
        plt.ylabel("P(OTHER | Feature)")
        plt.bar(bin_other[0][:-1],prob,width=np.diff(bin_other[0]),align='edge')
    plt.show()
    
    return bin_sum

def fa_graph_combine(vid_id_range,behavior,intervals=20):
    '''
    Binning for facing_angle
    '''
    
    lst_bins_mount = []
    lst_bins_attack = []
    lst_bins_other = []
    
    #do this for each video
    for i in range (vid_id_range[0],vid_id_range[1]):
        #find each behavior's bins
        #compute s_matrix for three diff classes
        mount = analyze_mount(str(i),data_train,data_test)
        attack = analyze_attack(str(i),data_train,data_test)
        other = analyze_other(str(i),data_train,data_test)
        
        #get frame duration of behaviors
        dur_mount = mount.filter_bout_seg()
        dur_attack = attack.filter_bout_seg()
        dur_other = other.get_other_frames()
        
        #compute facing_angle rate
        ma_mount = fa_mount(mount, dur_mount)
        ma_attack = fa_attack(attack, dur_attack)
        ma_other = fa_other(other,dur_other)
        
        #put them in separate lists
        lst_bins_mount.extend(ma_mount if ma_mount is not None else [])
        lst_bins_attack.extend(ma_attack if ma_attack is not None else[])
        lst_bins_other.extend(ma_other if ma_other is not None else [])

    #binning features from given videos
    bin_mount = s_matrix_bin(lst_bins_mount, intervals)
    bin_attack = s_matrix_bin(lst_bins_attack, intervals)
    bin_other = s_matrix_bin(lst_bins_other, intervals)
    
    bin_sum = []
    #take the same bin from all classes, compute probability
    #probability of the behavior given a value range
    for i in range(intervals):
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
            return print(f'No mount in videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_mount[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'MOUNT: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in facing_angle over 15 frames")
        plt.ylabel("P(MOUNT | Feature)")
        plt.bar(bin_mount[0][:-1],prob,width=np.diff(bin_mount[0]),align='edge') #equal width

    elif behavior == "attack":
        if bin_attack is None:
            return print(f'No attack in videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_attack[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'ATTACK: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in facing_angle over 15 frames")
        plt.ylabel("P(ATTACK | Feature)")
        plt.bar(bin_attack[0][:-1],prob,width=np.diff(bin_attack[0]),align='edge')
        
    else: #behavior == "other"
        for i in range(intervals):
            if len(bin_sum[i]) != 0:
                prob.append(len(bin_other[1][i])/len(bin_sum[i]))
            else:
                prob.append(0)
        #find how many are mount given a value range
        plt.title(f'OTHER: Videos {vid_id_range[0]}-{vid_id_range[1]-1}')
        plt.xlabel("change in facing_angle over 15 frames")
        plt.ylabel("P(OTHER | Feature)")
        plt.bar(bin_other[0][:-1],prob,width=np.diff(bin_other[0]),align='edge')
    plt.show()
    
    return bin_sum

def vc_graph_combine(vid_id_range,behavior,intervals,part):
    
    return None

#probability of the behavior given a value range
if __name__ == "__main__":
    
    start = time.time()

    #for i in range (1,10):   
    #    graph_s_matrix_bin(i, "attack")
    
    #y = graph_cr_bin(70,"attack")
    
    #x = cr_graph_combine([1,71],"mount",10)
    #z = fa_graph_combine([1,71],"other",10)
    
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



# x = raw Δsym_ratio per trial, y = mount/no-mount (0/1), one row per trial
X_linear = sm.add_constant(x)
model_linear = sm.Logit(y, X_linear).fit()

X_quad = sm.add_constant(np.column_stack([x, x**2]))
model_quad = sm.Logit(y, X_quad).fit()

# likelihood ratio test: does the quadratic term earn its extra parameter?
lr_stat = 2 * (model_quad.llf - model_linear.llf)
# compare lr_stat to chi-square with 1 df, or just compare model_linear.aic vs model_quad.aic

'''

    
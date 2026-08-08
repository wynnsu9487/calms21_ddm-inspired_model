#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 07:28:42 2026

@author: wynnsu
"""

import json
import os 
import numpy as np
import math
import matplotlib.pyplot as plt
import time

#Process and extract features from video coordinates
#Classes analyze_mount, analyze_attack

#NOTE: bouts are analyzed individually.
#This is more appropriate given that I want to distinct features between them.

#Redirect
os.chdir('/Users/wynnsu/Downloads/DeepLabCut/notebook_h5_csv')

#Open and load the JSON file
with open('calms21_task1_train.json', 'r') as file:
    data_train = json.load(file)
with open('calms21_task1_test.json','r') as file:
    data_test = json.load(file)

# =============================================================================
# Analyze Mount
# =============================================================================

class analyze_mount:
    '''
    An instance represents a video to be analyzed for mounting behaviors
    
    Instant attributes:
        vid_id: (str) video number/id
        data_train: (dict) xy coordinates from both mice, from all videos for training
        data_test:  (dict) xy coordinates from both mice, from all videosfor testing
        classific:  (dict) annotated classifications from all frames of all videos
        coor_intr:  (dict) coordinates from intruder
        coor_resi:  (dict) coordinates from resident
    '''
    
    # =============================================================================
    # Initializing & Processing
    # =============================================================================
    
    def __init__(self,vid_id,data_train,data_test):
        '''
        Initializes an instance of analyze_mount.

        Parameters:
            vid_id: (str) video number/id
            data_train: (dict) xy coordinates from both mice, from all videosfor training
            data_test:  (dict) xy coordinates from both mice, from all videosfor training
        '''
        
        #Modifies video id
        if len(vid_id) == 1:
            self.vid_id = '00'+vid_id
            
        elif len(vid_id) == 2:
            self.vid_id = '0'+vid_id
            
        self.data_train = data_train
        self.data_test = data_test
        
        #Converts dictionary to numpy array (and inverts it)
        if int(vid_id) <= 70:
            self.classific = self.conv_classific_n_coor("train")[0] #train videos
            self.coor_intr = self.invert_y(self.conv_classific_n_coor("train")[1])
            self.coor_resi = self.invert_y(self.conv_classific_n_coor("train")[2])
        
        else:
            self.classific = self.conv_classific("test")[0] #test videos
            self.coor_intr = self.invert_y(self.conv_classific_n_coor("test")[1])
            self.coor_resi = self.invert_y(self.conv_classific_n_coor("test")[2])
        
            
    def conv_classific_n_coor(self,data):
        '''
        Converts classification and xy coordinates from dictionary to numpy array 
        
        Parameter:
            data: (str) indicates videos dataset for either training or testing
        '''
        
        #stores classification and coordinates from a specific video
        if data == "train":
            np_classific = np.array(self.data_train['annotator-id_0']['task1/train/mouse'+self.vid_id+'_task1_annotator1']['annotations'])+1
            coor_intr    = np.array(self.data_train['annotator-id_0']['task1/train/mouse'+self.vid_id+'_task1_annotator1']['keypoints'])[:,0,:,:]
            coor_resi    = np.array(self.data_train['annotator-id_0']['task1/train/mouse'+self.vid_id+'_task1_annotator1']['keypoints'])[:,1,:,:]
        elif data == "test":
            np_classific = np.array(self.data_test['annotator-id_0']['task1/test/mouse'+self.vid_id+'_task1_annotator1']['annotations'])+1
            coor_intr    = np.array(self.data_test['annotator-id_0']['task1/test/mouse'+self.vid_id+'_task1_annotator1']['keypoints'])[:,0,:,:]
            coor_resi    = np.array(self.data_test['annotator-id_0']['task1/test/mouse'+self.vid_id+'_task1_annotator1']['keypoints'])[:,1,:,:]
        
        return (np_classific,coor_intr,coor_resi)

    def pre_mount(self):
        '''
        Identifies frames when there was investigation before mount
        '''
    
        lst_frame = []
        #Iterates the array
        for each in range (1,len(self.classific)):
            
            #Checks if there was an investigation happened before
            if self.classific[each-1] == 2 and self.classific[each] == 3:
                lst_frame.append(each)
        
        return np.array(lst_frame)
    
    def bout_end(self):
        '''
        Finds and returns frames when a mount bout ends
        '''
        
        lst_frame = []
        
        #Iterates the array
        for each in range(1,len(self.classific)):
            
            #Checks if investigation ended
            if self.classific[each-1] == 3 and self.classific[each] != 3:
                lst_frame.append(each)
        
        return np.array(lst_frame)
    
    def invert_y(self,np_coor):
        '''
        Inverts y-coordinates for more intuitive calculations
        
        Parameter:
            np_coor: (numpy array) coordinates from either intruder or resident
        '''
        np_coor[:,1,:] = np.abs(np_coor[:,1,:] - 570)
        
        return np_coor
    
    # =============================================================================
    # Helper functions
    # =============================================================================
    
    def centroid_all(self,frame,np_coor):
        '''
        Calculates centroids based on all body parts from one mouse
        
        Parameter:
            frame: (type) description
            np_coor: (type) description
        '''
        centroid_all_x = sum(np_coor[frame,0,:])/len(np_coor[frame,0,:])
        centroid_all_y = sum(np_coor[frame,1,:])/len(np_coor[frame,1,:])
    
        #Returns tuple with 2 elements, centroid's xy coordinate of a single frame
        return (centroid_all_x,centroid_all_y)

    def centroid_spec(self,frame,np_coor):
        '''
        Calculates centroids based on specific body parts
        
        Parameter:
            frame: (int) a specific frame in the video
            np_coor: (numpy array) coordinates for either resident or intruder
        '''
        centroid_x = sum(np_coor[:,frame,0])/len(np_coor[:,frame,0])
        centroid_y = sum(np_coor[:,frame,1])/len(np_coor[:,frame,1])
    
        #Returns tuple with 2 elements, centroid's xy coordinate of a single frame
        return (centroid_x,centroid_y)
    
    def dist(self,point1,point2):
        '''
        Calculates the distance between two points
        
        Parameter:
            point1: (list or numpy array) consists of xy coordinates
            point2: (list or numpy array) consists of xy coordinates
        '''
        x = point2[0] - point1[0]
        y = point2[1] - point1[1]
        
        return np.sqrt(x**2 + y**2)
    
    def dot_prod(self,v1,v2):
        '''
        Calculates dot product of two vectors
        
        Parameter:
            v1: (list) a vector with xy components
            v2: (list) a vector with xy components
        '''
        
        return v1[0]*v2[0] + v1[1]*v2[1]
    
    def angle_btw_lines(self,resi_nose,resi_neck,intr,add_intr=None):
        """
        Calculates the smallest angle (in degrees) between two lines;
        Each line has two points, with one point from each line possibly overlapping
        
        Parameter:
            resi_nose: (numpy array) coordinates of resident's nose
            resi_neck: (numpy array) coordinates of resident's neck
            intr:      (numpy array) coordinates from one of intruder's body part
            add_intr:  (numpy array) optional coordinates from another intruder's body part
            
        Three inputs: resident nose & neck & either intruder's centroid or hipL or hipR
        Four inputs: resident nose & neck AND intruder neck & centroid
        """
    
        if add_intr is None:
            #line ideal - consists of resi_nose AND intr
            dx1 = intr[0] - resi_nose[0]
            dy1 = intr[1] - resi_nose[1]
            
            #line real - consists of resi_nose AND resi_neck
            dx2 = resi_nose[0] - resi_neck[0]
            dy2 = resi_nose[1] - resi_neck[1]
        
            dot_product = self.dot_prod([dx1,dy1],[dx2,dy2])
        
            mag_ideal = np.sqrt(self.dot_prod([dx1,dy1],[dx1,dy1]))
            mag_real = np.sqrt(self.dot_prod([dx2,dy2],[dx2,dy2]))
            
            cos = dot_product / (mag_ideal*mag_real)
            angle = math.acos(dot_product/(mag_ideal*mag_real))
            angle_degree = math.degrees(angle)
                
            return angle_degree
    
        else:
            #line intruder
            dx1 = intr[0] - add_intr[0]
            dy1 = intr[1] - add_intr[1]
            
            #line resident
            dx2 = resi_nose[0] - resi_neck[0]
            dy2 = resi_nose[1] - resi_neck[1]
        
            dot_product = self.dot_prod([dx1,dy1],[dx2,dy2])
        
            mag_intr = np.sqrt(self.dot_prod([dx1,dy1],[dx1,dy1]))
            mag_resi = np.sqrt(self.dot_prod([dx2,dy2],[dx2,dy2]))
            
            cos = dot_product / (mag_intr*mag_resi)
            angle = math.acos(cos)
            angle_degree = math.degrees(angle)
                
            return angle_degree
        
    def count_match(self,match,arr):
        '''
        Counts how many elements in arr matches the string.
        
        Parameter:
            match: (str) any string (eg. True_Head) 
            arr:   (numpy array) matrix containing string
        '''
        
        count = 0
        row,col = np.shape(arr)
        
        for r in range(row):
            for c in range(col):
                if arr[r,c] == match:
                    count = count + 1
                    
        return count
        
    # =============================================================================
    # Main functions - extracting important features
    # =============================================================================
        
    def get_pre_onset(self,frame_of_interest):
        '''
        Return a list of 30 frames before a specific frame
        
        Parameter:
            frame_of_interest:  (int) a frame when behavior transition happens
        '''
        #Given a bout frame, go 30 frames before that bout.
        if frame_of_interest-30 < 0:
            return None
        return np.arange(frame_of_interest-30,frame_of_interest)
    
    def contact_region_numb(self,frame_of_interest,np_coor_resi,np_coor_intr):
        '''
        Calculates distance and angles betweeen specific body parts of two mice
        
        Parameter:
            frame_of_interest: (int) a frame when behavior transition happens
            np_coor_resi:      (numpy array) coordinates from resident
            np_coor_intr:      (numpy array) coordiantes from intruder
        '''
        
        resi_nose = np_coor_resi[frame_of_interest,:,0]
        resi_neck = np_coor_resi[frame_of_interest,:,3]
    
        intr_centroid_head = self.centroid_spec(frame_of_interest,np.array([np_coor_intr[:,:,0],np_coor_intr[:,:,1],np_coor_intr[:,:,2],np_coor_intr[:,:,3]]))
        intr_centroid_all =  self.centroid_all(frame_of_interest,np_coor_intr)
        intr_hipL = np_coor_intr[frame_of_interest,:,4]
        intr_hipR = np_coor_intr[frame_of_interest,:,5]
        intr_tail = np_coor_intr[frame_of_interest,:,6]
        
        nose_head = [self.dist(resi_nose,intr_centroid_head),self.angle_btw_lines(resi_nose,resi_neck,intr_centroid_head)]
        nose_hipL = [self.dist(resi_nose,intr_hipL),         self.angle_btw_lines(resi_nose,resi_neck,intr_hipL)]
        nose_hipR = [self.dist(resi_nose,intr_hipR),         self.angle_btw_lines(resi_nose,resi_neck,intr_hipR)]
        nose_cent = [self.dist(resi_nose,intr_centroid_all), self.angle_btw_lines(resi_nose,resi_neck,intr_centroid_all)]
        nose_tail = [self.dist(resi_nose,intr_tail),         self.angle_btw_lines(resi_nose,resi_neck,intr_centroid_all)]

        #array w/ shape (5,2), where rows are the 5 variables, columns consists of dist & angle
        return np.array([nose_head,nose_hipL,nose_hipR,nose_cent,nose_tail])
        
    
    #contact region relative to intruder - where resident's head is oriented on intruder's body
    def contact_region(self,frame_of_interest,np_coor_resi,np_coor_intr):
        '''
        Checks whether resident is facing & in proximity with a body part of intruder;
        Returns elements {True_Head,True_Centroid,True_Hip,True_Tail,None}
        
        Parameter:
            frame_of_interest: (int) a frame when behavior transition happens
            np_coor_resi:      (numpy array) coordinates from resident
            np_coor_intr:      (numpy array) coordiantes from intruder
        '''
        resi_nose = np_coor_resi[frame_of_interest,:,0]
        resi_neck = np_coor_resi[frame_of_interest,:,3]
    
        intr_centroid_head = self.centroid_spec(frame_of_interest,np.array([np_coor_intr[:,:,0],np_coor_intr[:,:,1],np_coor_intr[:,:,2],np_coor_intr[:,:,3]]))
        intr_centroid_all =  self.centroid_all(frame_of_interest,np_coor_intr)
        intr_hipL = np_coor_intr[frame_of_interest,:,4]
        intr_hipR = np_coor_intr[frame_of_interest,:,5]
        intr_tail = np_coor_intr[frame_of_interest,:,6]
    
        #if resident nose is close to intruder head
        if (self.dist(resi_nose,intr_centroid_head) <= 40 and self.angle_btw_lines(resi_nose,resi_neck,intr_centroid_head) <= 15):
            return "True_Head"
        
        #elif nose is close to hip AND orientation (from neck to head) is to hip
        elif ((self.dist(resi_nose,intr_hipL) <= 40 and self.angle_btw_lines(resi_nose,resi_neck,intr_hipL) <= 15) or 
              (self.dist(resi_nose,intr_hipR) <= 40 and self.angle_btw_lines(resi_nose,resi_neck,intr_hipR) <= 15)
             ):
            return "True_Hip"
                
        #elif orientation is close to centroid
        elif self.dist(resi_nose,intr_centroid_all) <= 40 and self.angle_btw_lines(resi_nose,resi_neck,intr_centroid_all) <= 15:
            return "True_Centroid"
    
        #elif orientation is close to tail
        elif (self.dist(resi_nose,intr_tail) <= 40 and self.angle_btw_lines(resi_nose,resi_neck,intr_tail) <= 15):
            return "True_Tail"
    
        #else False
        else:
            return "None"
    
    #s-matrix modified from Khan et al 2025 - quantifies socio-spatial arrangement of body parts (ie head-body-tail interaction relative to other)
    '''
    S-matrices (symmetry ratio ≈ 0) indicate balanced head-body-tail
    (HBT) interaction, e.g., M1 and M2 maintain similar distances between
    their heads, bodies, and tails, whereas asymmetric S-matrices (symmetry
    ratio ≈ 1) indicate disbalance.
    '''
    def s_matrix(self,frame_of_interest,np_coor_resi,np_coor_intr):
        '''
        Calculates symmetry ratio, given coordinates of two mice.
        
        S-matrices (symmetry ratio ≈ 0) indicate balanced head-body-tail
        (HBT) interaction, e.g., mice maintain similar distances between HBT,
        whereas asymmetric S-matrices (symmetry ratio ≈ 1) indicate disbalance.
        
        Parameter:
            frame_of_interest: (int) a frame when behavior transition happens
            np_coor_resi:      (numpy array) coordinates from resident
            np_coor_intr:      (numpy array) coordiantes from intruder
        '''
    
        #0: nose-ears-neck centroids (or only nose)
        #1: centroid of all body parts
        #2: tail base
        #construct list of body parts for each mouse
    
        head_resi = self.centroid_spec(frame_of_interest,np.array([np_coor_resi[:,:,0],np_coor_resi[:,:,1],np_coor_resi[:,:,2],np_coor_resi[:,:,3]]))
        head_intr = self.centroid_spec(frame_of_interest,np.array([np_coor_intr[:,:,0],np_coor_intr[:,:,1],np_coor_intr[:,:,2],np_coor_intr[:,:,3]]))
        
        lst_resi = [head_resi,self.centroid_all(frame_of_interest,np_coor_resi),np_coor_resi[frame_of_interest,:,6]]
        lst_intr = [head_intr,self.centroid_all(frame_of_interest,np_coor_intr),np_coor_intr[frame_of_interest,:,6]]
    
        #construct matrix (3x3), where each element represents distance between pair of body parts
        s_m = np.zeros((3,3))
        row,col = np.shape(s_m)
    
        #iterate rows and columns
        for r in range(row):
            for c in range(col):
                #compute distance between the two body parts in iteration
                d = self.dist(lst_resi[r],lst_intr[c])
                s_m[r,c] = d
    
        diff_m = s_m - np.transpose(s_m)
    
        ori_m_frobenius = np.sqrt(np.sum(np.square(s_m)))
        diff_m_frobenius = np.sqrt(np.sum(np.square(diff_m)))
        
        sym_ratio = diff_m_frobenius/ori_m_frobenius
    
        return sym_ratio
    
    def facing_angle(self,frame_of_interest,np_coor_resi,np_coor_intr):
        '''
        Returns angle between resident's heading direction & intruder's centroid
        
        Parameter:
            frame_of_interest: (int) a frame when behavior transition happens
            np_coor_resi:      (numpy array) coordinates from resident
            np_coor_intr:      (numpy array) coordiantes from intruder
        '''
        #coordinates during the frame of interest
        fr_resi_nose = np_coor_resi[frame_of_interest,:,0]
        fr_resi_neck = np_coor_resi[frame_of_interest,:,3]
        fr_intr_neck = np_coor_intr[frame_of_interest,:,3]
        fr_intr_centroid = np_coor_intr[frame_of_interest,:,6]
    
        #get angle btw resident's heading direction & intruder's centroid
        angle = self.angle_btw_lines(fr_resi_nose,fr_resi_neck,fr_intr_neck,fr_intr_centroid)
    
        return angle
    
    def time_since_last_bout(self,prev_bout,curr_frame):
        '''
        Returns how long since the last bout has ended
    
        Hypothesis: time since last bout predicts escalation
        (ie Time-since-last-bout as noise term)
        
        Parameter:
            prev_bout:  (int) the frame when previous bout ends
            curr_frame: (int) the current
        '''
        return curr_frame - prev_bout
    
    def visual_cone(self,frame_of_interest,np_coor_resi,np_coor_intr): 
        '''
        Indicates if intruder is within resident ±45° cone (like a vision field)
        
        Parameter:
            frame_of_interest: (int) a frame when behavior transition happens
            np_coor_resi:      (numpy array) coordinates from resident
            np_coor_intr:      (numpy array) coordiantes from intruder
        '''
        #key parts
        intr_centroid_head = self.centroid_spec(frame_of_interest,np.array([np_coor_intr[:,:,0],np_coor_intr[:,:,1],np_coor_intr[:,:,2],np_coor_intr[:,:,3]]))
        intr_centroid_body = self.centroid_spec(frame_of_interest,np.array([np_coor_intr[:,:,3],np_coor_intr[:,:,4],np_coor_intr[:,:,5],np_coor_intr[:,:,6]]))
        resi_nose = np_coor_resi[frame_of_interest,:,0]
        resi_neck = np_coor_resi[frame_of_interest,:,3]
    
        med_line = resi_nose - resi_neck         #median line of resident
        med_line_norm = np.linalg.norm(med_line) #distance
    
        vec_to_head = intr_centroid_head - resi_nose
        vec_to_body = intr_centroid_body - resi_nose
    
        #distance from resident's nose to intruder's head & body
        head_dist = np.linalg.norm(vec_to_head)
        body_dist = np.linalg.norm(vec_to_body)
    
        if head_dist > 0:
            cos_head = self.dot_prod(med_line,vec_to_head) / (med_line_norm*head_dist)
            
        if body_dist > 0:
            cos_body = self.dot_prod(med_line,vec_to_body) / (med_line_norm*body_dist)
            
        return (cos_head,cos_body)
    
    # =============================================================================
    # Display distributions
    # =============================================================================
    
    def stats_arr(self,graph=True):
        '''
        Calculates all main features
        Optionally graphs the features as a function of 30 frame 
        
        Parameter:
            graph: (bool) shows feature graphs or not.
        '''

        #for each frame in gap, store all 30 frames before the frame in a list -> get_pre_onset ()
        lst_frames_vid = []
        mount_gap = self.pre_mount()
        for each in mount_gap:
            frames_vid = self.get_pre_onset(each)
            lst_frames_vid.append(frames_vid)
        
        lst_bout_end = self.bout_end() #get frames when bout ends
        
        #for each of those 30 frames, run contact_region(), s_matrix(), facing_angle()
        vid_stats = np.zeros((6,len(lst_frames_vid),30),dtype=object)
        sec_ind = 0
        
        #loop through transition frame
        for lst in range(len(lst_frames_vid)):
            thi_ind = 0
            #loop through pre-transition frames
            for ele in lst_frames_vid[lst]:
                c_r = self.contact_region(ele,self.coor_resi,self.coor_intr)   #contact_region()
                s_m = self.s_matrix(ele,self.coor_resi,self.coor_intr)         #s_matrix()
                f_a = self.facing_angle(ele,self.coor_resi,self.coor_intr)     #facing_angle()
                if lst == 0:
                    t_l = self.time_since_last_bout(0,ele)                     #time_since_last_bout()
                else:
                    t_l = self.time_since_last_bout(lst_bout_end[lst-1]+1,ele) #time_since_last_bout()
                v_c_h = self.visual_cone(ele,self.coor_resi,self.coor_intr)[0] #visual_cone() - head
                v_c_b = self.visual_cone(ele,self.coor_resi,self.coor_intr)[1] #visual_cone() - body
        
                vid_stats[0,sec_ind,thi_ind] = c_r
                vid_stats[1,sec_ind,thi_ind] = s_m
                vid_stats[2,sec_ind,thi_ind] = f_a
                vid_stats[3,sec_ind,thi_ind] = t_l
                vid_stats[4,sec_ind,thi_ind] = v_c_h
                vid_stats[5,sec_ind,thi_ind] = v_c_b
                thi_ind = thi_ind + 1
            sec_ind = sec_ind + 1
            
        if graph == True:
        
            ###display distributions across those 30 frames 
            
            #contact_region()
            plt.figure(1)
            #x = ["True_Head","True_Hip","True_Centroid","True_Tail","None"]
            x = ["True_Head","True_Hip","True_Centroid","True_Tail"] #TODO change this (or not)
            y = np.zeros(4)
            
            for i in range(len(y)):
                y[i] = self.count_match(x[i],vid_stats[0])  
            plt.title(f'MOUNT: contact region, video {self.vid_id}')
            plt.bar(x,y)
            plt.show()
            
            #s_matrix()
            plt.figure(2)
            for i in range(len(lst_frames_vid)): 
                plt.plot(np.arange(0,30),vid_stats[1,i,:],label=f'frame {mount_gap[i]}')
            
            plt.title(f'MOUNT: pre-transition frames vs. sym_ratio, video {self.vid_id}')
            #plt.legend(fontsize=7,loc='upper left')
            plt.show()
            
            #facing_angle()
            plt.figure(3)
            for i in range(len(lst_frames_vid)):
                plt.plot(np.arange(0,30),vid_stats[2,i,:],label=f'frame {mount_gap[i]}')
            plt.title(f'MOUNT: pre-transition frames vs. facing angle, video {self.vid_id}')
            #plt.legend(fontsize=7,loc='upper left')
            plt.show()
            
            #time_since_last_bout()
            plt.figure(4)
            plt.hist(vid_stats[3,:,-1])
            plt.title(f'MOUNT: freuqnecy distribution of time since last bout, video {self.vid_id}')
            plt.show()
            
            #visual_cone()
            plt.figure(5)
            threshold = np.cos(np.deg2rad(45))
            plt.axhline(y=threshold,color='r',linestyle='--',label='45° cone')
            
            for i in range(len(lst_frames_vid)):
                plt.plot(np.arange(0,30),vid_stats[4,i,:],color='b',label=f'frame {mount_gap[i]}')
                plt.plot(np.arange(0,30),vid_stats[5,i,:],color='g')
            plt.ylim(-1.05, 1.05)
            plt.xlabel("Frame")
            plt.ylabel("Cosine similarity")
            plt.title(f'MOUNT: pre-transition frames vs. visual cones, video {self.vid_id}')
            #plt.legend(fontsize=7,loc='lower left')
            plt.show()
        
        return vid_stats

# =============================================================================
# Analyze Attack - specifically displays feature distributions for attack
# Inherited from class analyze_mount
# =============================================================================

class analyze_attack(analyze_mount):
    '''
    analyze_attack calculates features before attack bouts
    '''
    
    def __init__(self,vid_id,data_train,data_test):
        '''
        Initializes an instance of analyze_mount.

        Parameters:
            vid_id: (str) video number/id
            data_train: (dict) xy coordinates from both mice, from all videosfor training
            data_test:  (dict) xy coordinates from both mice, from all videosfor training
        '''
        
        super().__init__(vid_id,data_train,data_test)
        
    def pre_attack(self):
        '''
        Identifies frames when there was investigation before mount
        '''
        lst_frame = []
        
        #iterate the array
        for each in range (1,len(self.classific)):
            #for every segment of mounting, check if there was an investigation happened before
            if self.classific[each-1] == 2 and self.classific[each] == 1:
                lst_frame.append(each)
        
        return np.array(lst_frame)
    
    # =============================================================================
    # Display dsibutions
    # =============================================================================
    
    def stats_arr(self,graph=True):
        '''
        Calculates all main features
        Optionally graph the features as a function of 30 frame 
        
        Parameter:
            graph: (bool) shows feature graphs or not.
        '''
        
        #for each frame in gap, store all 30 frames before the frame in a list -> get_pre_onset ()
        lst_frames_vid = []
        attack_gap = self.pre_attack()
        for each in attack_gap:
            frames_vid = self.get_pre_onset(each)
            lst_frames_vid.append(frames_vid)
        
        lst_bout_end = self.bout_end() #get frames when bout ends
        
        #for each of those 30 frames, run contact_region(), s_matrix(), facing_angle()
        vid_stats = np.zeros((6,len(lst_frames_vid),30),dtype=object)
        sec_ind = 0
        
        #loop through transition frame
        for lst in range(len(lst_frames_vid)):
            thi_ind = 0
            #loop through pre-transition frames
            for ele in lst_frames_vid[lst]:
                c_r = self.contact_region(ele,self.coor_resi,self.coor_intr)   #contact_region()
                s_m = self.s_matrix(ele,self.coor_resi,self.coor_intr)         #s_matrix()
                f_a = self.facing_angle(ele,self.coor_resi,self.coor_intr)     #facing_angle()
                if lst == 0:
                    t_l = self.time_since_last_bout(0,ele)                     #time_since_last_bout()
                else:
                    t_l = self.time_since_last_bout(lst_bout_end[lst-1]+1,ele) #time_since_last_bout()
                v_c_h = self.visual_cone(ele,self.coor_resi,self.coor_intr)[0] #visual_cone() - head
                v_c_b = self.visual_cone(ele,self.coor_resi,self.coor_intr)[1] #visual_cone() - body
        
                vid_stats[0,sec_ind,thi_ind] = c_r
                vid_stats[1,sec_ind,thi_ind] = s_m
                vid_stats[2,sec_ind,thi_ind] = f_a
                vid_stats[3,sec_ind,thi_ind] = t_l
                vid_stats[4,sec_ind,thi_ind] = v_c_h
                vid_stats[5,sec_ind,thi_ind] = v_c_b
                thi_ind = thi_ind + 1
            sec_ind = sec_ind + 1
        
        if graph == True:
        
            ###display distributions across those 30 frames 
            
            #contact_region()
            plt.figure(1)
            x = ["True_Head","True_Hip","True_Centroid","True_Tail","None"] #TODO reminder to possibly change this (or not)
            y = np.zeros(5)
            
            for i in range(len(y)):
                y[i] = self.count_match(x[i],vid_stats[0])  
            plt.title(f'ATTACK: contact region, video {self.vid_id}')
            plt.bar(x,y)
            plt.show()
            
            #s_matrix()
            plt.figure(2)
            for i in range(len(lst_frames_vid)): 
                plt.plot(np.arange(0,30),vid_stats[1,i,:],label=f'frame {attack_gap[i]}')
            
            plt.title(f'ATTACK: pre-transition frames vs. sym_ratio, video {self.vid_id}')
            #plt.legend(fontsize=7,loc='upper left')
            plt.show()
            
            #facing_angle()
            plt.figure(3)
            for i in range(len(lst_frames_vid)):
                plt.plot(np.arange(0,30),vid_stats[2,i,:],label=f'frame {attack_gap[i]}')
            plt.title(f'ATTACK: pre-transition frames vs. facing angle, video {self.vid_id}')
            #plt.legend(fontsize=7,loc='upper left')
            plt.show()
            
            #time_since_last_bout()
            plt.figure(4)
            plt.hist(vid_stats[3,:,-1])
            plt.title(f'ATTACK: freuqnecy distribution of time since last bout, video {self.vid_id}')
            plt.show()
            
            #visual_cone()
            plt.figure(5)
            threshold = np.cos(np.deg2rad(45))
            plt.axhline(y=threshold,color='r',linestyle='--',label='45° cone')
            
            for i in range(len(lst_frames_vid)):
                plt.plot(np.arange(0,30),vid_stats[4,i,:],color='b',label=f'frame {attack_gap[i]}')
                plt.plot(np.arange(0,30),vid_stats[5,i,:],color='g')
            plt.ylim(-1.05, 1.05)
            plt.xlabel("Frame")
            plt.ylabel("Cosine similarity")
            plt.title(f'ATTACK: pre-transition frames vs. visual cones, video {self.vid_id}')
            #plt.legend(fontsize=7,loc='lower left')
            plt.show()
        
        return vid_stats
            
if __name__ == "__main__":

    '''
    start = time.time()
    for i in range(1,5):
        vid_mount = analyze_mount(str(i),data_train,data_test)
        #vid_attack = analyze_attack(str(i),data_train,data_test)
        
        vid_mount.stats_arr()
        #vid_attack.stats_arr()
        print(f'Done w/ {i}')
    
    end = time.time()
    print(f'ALL DONE! Took {end-start} seconds')
    '''
    
    
    
    
'''
This code is licensed and documented by Teuvonet Technologies. 
Any use of this code, proprietary or personal, needs approval from the company.

'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import time
import imp
from pandas import *
#import pyopencl as cl
#import pyopencl.algorithm as algo
#import pyopencl.array as pycl_array
import sklearn.metrics as met
from collections import defaultdict
import distEncode
from Classification_MasterCode import commenter
import copy
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import ctypes

def phase2(target_arr, NET_LATTICE, LEARNING_RATES_PHASE_2, Input_Data, FeatureRankingByClass, NUM_CLASSES,
 Max_FeatureSpaces_Size, no_data_Passes_kernel_Phase_2, NUM_DATASET_PASSES, 
 CHUNK_SIZE, train_crossvalidation_percentage, New_Feature_Order, list_num_feature_featurespaces, 
 total_num_features_all_featurespaces, cumulative_no_features_featurespace):

        #del context
	commenter("Phase 2 Details")
        os.environ['PYOPENCL_NO_CACHE'] = '1'
	#Intial Parameters
	NUM_NEURONS_PER_NET = [ i * i for i in NET_LATTICE] # [9,16,25]
	TOTAL_NUM_NEURONS = sum(NUM_NEURONS_PER_NET) # 3*3 + 4*4 + 5*5 = 50 neurons
	NUM_NETS = len(NET_LATTICE) #[3,4,5] = 3
	NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE_2) # Number of Learning Rates
	NUM_DATAPOINTS = Input_Data.shape[0] #Number of tuples/datapoints
	SIZE_OF_PARTITIONS = 100 #Maximum Number of Feature in a Partition



	TOTAL_NUM_FEATURES = len(New_Feature_Order) #New Feature Order
	NUM_PARTITIONS = len(list_num_feature_featurespaces)


	

	print("\n")
	# print("Now doing Phase 2 Details Below\n")
	print("Number of Datapoints :"+str(NUM_DATAPOINTS))
	print("Number of classes in the given data :"+str(NUM_CLASSES))
	print("Net_lattices : " + str(NET_LATTICE))
	print("Total Number of features : " + str(TOTAL_NUM_FEATURES))
	print("Total Number of learning rates : " + str(NUM_LEARNING_RATES))
	print("Number of Partitions : "+str(NUM_PARTITIONS))
	print("Number of Passes of Data in kernel : {}".format(no_data_Passes_kernel_Phase_2))
	print("total num of features : {}".format(total_num_features_all_featurespaces))
	print("cumulative_no_features_featurespace : {}".format(cumulative_no_features_featurespace))
	print("list_num_feature_featurespaces : {}".format(list_num_feature_featurespaces))
	#print("New Feature Order for Phase2 : {}".format(New_Feature_Order))
        print("shape of new feature_space",np.shape(New_Feature_Order))
	print("\n")
	phase_2_training = time.time()
 	# KERNEL BUFFER ARRAYS
	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	myFloatType = np.float32  # Global Float Type for Kernel
	myIntType = np.int32
	#Input Data
	Input_Data = Input_Data.iloc[:,New_Feature_Order]

	#Distance map array stores the distance between neuron center and Input data of every feature
	distance_base_map = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS ), dtype = myFloatType)
	
    #    distance_base_map_buf_1 =  cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = distance_base_map)#write buffer
	#Base Map will have the neuron centers. Size = NoOfneuron * NoOfFeatures
	map_data_size = NUM_LEARNING_RATES * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces	
	map_data = np.random.uniform(low=0, high=0, size=(map_data_size,)).astype(myFloatType)

    #    map_data_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = map_data)#write buffer
	#Number of features in each Partition
	NUM_FEATURES_IN_PARTITIONS = np.array(list_num_feature_featurespaces, dtype =  myIntType)

    #    NUM_FEATURES_IN_PARTITIONS_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_FEATURES_IN_PARTITIONS)#read buffer
	#learning rates for phase1
	LEARNING_RATES_PHASE_2 = np.array(LEARNING_RATES_PHASE_2, dtype =  myFloatType)

	#Cumulative Number of features 
	cumulative_no_features_featurespace = np.array(cumulative_no_features_featurespace, dtype = myIntType)

    #    cumulative_no_features_featurespace_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_features_featurespace)
	#Number of Neurons in each nettures
	NUM_NEURONS_PER_NET = np.array(NUM_NEURONS_PER_NET, dtype = myIntType)

    #    NUM_NEURONS_PER_NET_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_NEURONS_PER_NET)#read buffer
	#Array to store minimum distance neuron for each net.
	min_dist_pos = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myIntType)
   
    #    min_dist_pos_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = min_dist_pos)#write buffer
	#Array to store minimum distance position of a neuron in each net
	min_dist_array = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myFloatType)

    #    min_dist_array_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = min_dist_array)#write buffer
	#NeighBourRate Array
	neigh_rate = np.array(np.ones(NUM_NETS * NUM_PARTITIONS * NUM_LEARNING_RATES), dtype = myFloatType) 
	new_neigh_rate = copy.deepcopy(neigh_rate)

    #    new_neigh_rate_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = new_neigh_rate)#write buffer
	new_learning_rate_list = copy.deepcopy(LEARNING_RATES_PHASE_2)

    #    new_learning_rate_list_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = new_learning_rate_list)#read buffer

	# cumulative weight change
	cumulative_weight_change_per_neuron = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS), dtype = myFloatType)
        
    #    cumulative_weight_change_per_neuron_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_weight_change_per_neuron)#write buffe#write buffer
	#This array is helpful for indexing of the Input data at kernel
	InputIndex_Per_Partition_Size = len(cumulative_no_features_featurespace)
	InputIndex_Per_Partition = np.array(np.zeros(InputIndex_Per_Partition_Size), dtype = myIntType)
    #    InputIndex_Per_Partition_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = InputIndex_Per_Partition)#read buffer

	#---------------------------------------------------------------------------------------------------------------------------------------------------------
        TOTAL_NUM_NEURONS = np.array(TOTAL_NUM_NEURONS, dtype = myIntType)
        NUM_NETS = np.array(NUM_NETS, dtype = myIntType)
        
        NUM_PARTITIONS = np.array(NUM_PARTITIONS, dtype = myIntType)
        TOTAL_NUM_FEATURES = np.array(TOTAL_NUM_FEATURES, dtype = myIntType)
        no_data_Passes_kernel_Phase_2 = np.array(no_data_Passes_kernel_Phase_2, dtype = myIntType)
        total_num_features_all_featurespaces = np.array(total_num_features_all_featurespaces, dtype = myIntType)
        NET_LATTICE = np.array(NET_LATTICE, dtype = myIntType)
    #    NET_LATTICE_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NET_LATTICE)#read buffer
	#Define the sizes of the blocks and grids
	grid_X, grid_Y = NUM_PARTITIONS*TOTAL_NUM_NEURONS, NUM_LEARNING_RATES
	#workUnits_X, workUnits_Y, workUnits_Z = TOTAL_NUM_NEURONS, 1, 1

	#phase2_time = time.time()
	for l in range(NUM_DATASET_PASSES):
		for learning_rate_Index in range(0, NUM_LEARNING_RATES):
			print("learning_rate_Index", learning_rate_Index)
			thread_id = 0 + NUM_PARTITIONS * TOTAL_NUM_NEURONS * learning_rate_Index
			for j in range(0, NUM_PARTITIONS):
				for k in range(0, TOTAL_NUM_NEURONS):
					print("k ,thread_id", k, thread_id)
				# print("Dataset Pass ", l+1)
					sum_cum_wt_change = 0
					for i in range(0, len(Input_Data), CHUNK_SIZE):
						chunk_inp = Input_Data.iloc[i:i + CHUNK_SIZE, :]  # extract the input data with the row number.
						Final_Num_DataPoints = min(len(Input_Data), len(chunk_inp))
						Final_Num_DataPoints = np.array(Final_Num_DataPoints, dtype=np.int32)
					# print("Final_Num_DataPoints", Final_Num_DataPoints)
					# Buffer array to hold the final number of datapoints
					# Final_Num_DataPoints_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = Final_Num_DataPoints)
					# Define the sizes of the blocks and grids
						grid_X, grid_Y = NUM_PARTITIONS * TOTAL_NUM_NEURONS, 1
					# workUnits_X, workUnits_Y, workUnits_Z = , 1, 1
						chunk_inp = chunk_inp.values.ravel()  # 1D array of the input of 5 chunks.
						chunk_inp = np.array(chunk_inp, dtype=myFloatType)  # read
					# chunk_inp = ctypes.c_float()
					# chunk_inp_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,hostbuf=chunk_inp)  # read buffer
						print("thread_id", thread_id)
						feature_space_blockID = j
						fun = ctypes.CDLL("/home/asim/SERIAL_CPU/Streaming_OPENCL_Classification/trainingNeurons.so")
						fun.trainingNeurons.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
														ctypes.c_int, ctypes.c_int, ctypes.c_int,
														np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
														np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
														ctypes.c_int, ctypes.c_int, ctypes.c_int]
						returnVale = fun.trainingNeurons(TOTAL_NUM_NEURONS,
														 NUM_NETS,
														 Final_Num_DataPoints,
														 NUM_PARTITIONS,
														 TOTAL_NUM_FEATURES,
														 total_num_features_all_featurespaces,
														 no_data_Passes_kernel_Phase_2,
														 NUM_FEATURES_IN_PARTITIONS,
														 cumulative_no_features_featurespace,
														 cumulative_weight_change_per_neuron,
														 NUM_NEURONS_PER_NET,
														 map_data,
														 min_dist_pos,
														 min_dist_array,
														 chunk_inp,
														 distance_base_map,
														 new_neigh_rate,
														 new_learning_rate_list,
														 NET_LATTICE,
														 InputIndex_Per_Partition,
														 thread_id,
														 feature_space_blockID,
														 learning_rate_Index
														 )

						print("returnVale", returnVale)
						print("map_data", map_data)
					thread_id = thread_id + 1
					print("iterartion over")

		if (l % 2 == 0):
			new_neigh_rate = np.array([lr * 0.9 for lr in new_neigh_rate], dtype=np.float32)
			new_learning_rate_list = np.array([lr * 0.9 for lr in new_learning_rate_list], dtype=np.float32)


    	phase_2_training_end = time.time()
    	time_phase_2 = phase_2_training_end - phase_2_training
     	print("time_phase_2", time_phase_2)


	#ACTIVE CENTERS
	##################################################################################################################################################

	#Reading the kernel File
	myFloatType = np.float32 #Global Float Type for Kernel
	myIntType = np.int32 #Global int Type for Kernel



	cumulative_no_neurons_per_net = []
	cumulative = 0
	cumulative_no_neurons_per_net.append(0)
	for val in NUM_NEURONS_PER_NET:
		cumulative += val
		cumulative_no_neurons_per_net.append(val)


	#KERNEL BUFFER ARRAYS
	#----------------------------------------------------------------------------------------------------------------------
	Input_Data = Input_Data.values.ravel() #Taking Input in the order of shuffled columns
	Input_Data = np.array(Input_Data, dtype = myFloatType) #Changing to a single dimensional array
    #    Input_Data_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = Input_Data)#read buffer
	#Cumulative Array of features in each partition array
	cumulative_no_features_featurespace = np.array(cumulative_no_features_featurespace, dtype= myIntType)
        
    #    cumulative_no_features_featurespace_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_features_featurespace)#read buffer
	#Cumulative number of neurons per net
	cumulative_no_neurons_per_net = np.array(cumulative_no_neurons_per_net, dtype= myIntType)

    #    cumulative_no_neurons_per_net_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_neurons_per_net)#read buffer#read buffer
	#Buffer Array to store cumulative sum of distances from each neuron to its feature
	distance_map = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS), dtype = myFloatType)
  
    #    distance_map_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = distance_map)#read buffer
	#Converting Target array to int
	target_arr_int = np.array(target_arr.values.ravel(), dtype=np.int32)
        
    #    target_arr_int_buf_2 =  cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = target_arr_int)#read buffer
	#Array to store minimum distance neuron for each net.
	min_pos = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myIntType)

    #    min_pos_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_pos)#read buffer
	#Array to store minimum distance position of a neuron in each net
	min_array = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myFloatType)
     
    #    min_array_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_array)#read buffer
	#Array to store how many datapoints for each class for each neuron
	active_centers_per_class = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS * NUM_CLASSES,)).T, dtype = myIntType)

    #    active_centers_per_class_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_per_class)#write buffer
	#Arrayto store dominating class for each neuron
	active_centers_dominant_class = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS,)).T, dtype = myIntType)
 
    #    active_centers_dominant_class_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_dominant_class)#write buffer
	#Active Centers Total Count
	active_centers_total_count = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS,)).T, dtype = myIntType)
 
    #    active_centers_total_count_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_total_count)#write buffer
	#Active Centers Total Count
	active_dominant_class_count = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS,)).T, dtype = myIntType)

    #    active_dominant_class_count_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_dominant_class_count)#write buffer
	#Number of features in each partition
	NUM_FEATURES_PER_PARTITION = np.array(list_num_feature_featurespaces, dtype = myIntType)

    #    map_data_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = map_data)#read buffer

    #    Input_Data_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = Input_Data)#read buffer

    #    InputIndex_Per_Partition_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = InputIndex_Per_Partition)#read buffer
    #    NUM_NEURONS_PER_NET_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_NEURONS_PER_NET)#read buffer
    #    NUM_FEATURES_PER_PARTITION_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_FEATURES_PER_PARTITION)

        # conversion to fp32 from argument of phase2
        NUM_DATAPOINTS = np.array(NUM_DATAPOINTS, dtype= myIntType)
        NUM_CLASSES = np.array(NUM_CLASSES, dtype = myIntType)

	#Define the sizes of the blocks and grids
	#grid_X, grid_Y, = NUM_PARTITIONS, NUM_LEARNING_RATES, 1
	#noofThreads = int(TOTAL_NUM_NEURONS) #CUDA Doesn't allow np.int64 variables so doing a typecase to int
	#workUnits_X, workUnits_Y, workUnits_Z = noofThreads, 1, 1
        
	activeCentersTime = time.time()
	# print(active_centers)

	for learning_rate_Index in range(0, NUM_LEARNING_RATES):
		print("learning_rate_Index", learning_rate_Index)
		thread_id = 0 + NUM_PARTITIONS * TOTAL_NUM_NEURONS * learning_rate_Index
		for j in range(0, NUM_PARTITIONS):
			for k in range(0, TOTAL_NUM_NEURONS):
				print("k ,thread_id", k, thread_id)
				print("thread_id", thread_id)
				feature_space_blockID = j
				active_center_cal = ctypes.CDLL(
				"/home/asim/SERIAL_CPU/Streaming_OPENCL_Classification/activecenters.so")
				active_center_cal.activecenters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
																ctypes.c_int, ctypes.c_int, ctypes.c_int,
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.float32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.float32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.float32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.float32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),

																ctypes.c_int, ctypes.c_int, ctypes.c_int]
				returnVale = active_center_cal.activecenters(TOTAL_NUM_FEATURES,
																 NUM_PARTITIONS,
																 NUM_DATAPOINTS,
																 TOTAL_NUM_NEURONS,
																 NUM_NETS,
																 NUM_CLASSES,
																 total_num_features_all_featurespaces,
																 NUM_FEATURES_PER_PARTITION,
																 cumulative_no_features_featurespace,
																 cumulative_no_neurons_per_net,
																 Input_Data,
																 map_data,
																 distance_map,
																 target_arr_int,
																 min_pos,
																 min_array,
																 active_centers_per_class,
																 active_centers_dominant_class,
																 active_centers_total_count,
																 active_dominant_class_count,
																 NUM_NEURONS_PER_NET,
																 InputIndex_Per_Partition,
																 thread_id,
																 feature_space_blockID,
																 learning_rate_Index)

				print("returnVale", returnVale)
				thread_id = thread_id + 1
				print("active_centers_per_class ", active_centers_per_class)
				print("active_centers_dominant_class ", active_centers_dominant_class)
				print("active_centers_total_count ", active_centers_total_count)


# map_data =map_data+map_data


	print("\nTime for Active Centers phase 2 : {0:.5f} secs".format(time.time()- activeCentersTime))

	#CALCULATE RADIUS
	##################################################################################################################################################

	#Reading the kernel File

	myFloatType = np.float32 #Global Float Type for Kernel
	myIntType = np.int32 #Global int Type for Kernel
	

	#KERNEL BUFFER ARRAYS
	#----------------------------------------------------------------------------------------------------------------------
	radius_map = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS,)).T, dtype = myFloatType)
    #    radius_map_buf_3 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = radius_map)#write buffer
	#Array to store minimum distance neuron for each net.
	min_pos = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myIntType)
    #    min_pos_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_pos)#read buffer#read buffer
	#Array to store minimum distance position of a neuron in each net
	min_array = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myFloatType)
    #    min_array_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_array)#read buffer

	#Buffer Array to store cumulative sum of distances from each neuron to its feature
	distance_map = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS), dtype = myFloatType)
    #    distance_map_buf_3 =cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = distance_map)#read buffer


    #    NUM_FEATURES_PER_PARTITION_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_FEATURES_PER_PARTITION)
    #    cumulative_no_features_featurespace_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_features_featurespace)#read buffer
    #    NUM_NEURONS_PER_NET_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_NEURONS_PER_NET)#read buffer
    #    active_centers_dominant_class_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_dominant_class)#read buffer
    #    map_data_buf_3 =  cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = map_data)#read buffer
    #    Input_Data_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = Input_Data)#read buffer
    #    InputIndex_Per_Partition_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = InputIndex_Per_Partition)#read buffer

	#grid_X, grid_Y, grid_Z = NUM_PARTITIONS, NUM_LEARNING_RATES, 1
	#noofThreads = int(TOTAL_NUM_NEURONS) #CUDA Doesn't allow np.int64 variables so doing a typecase to int
	#workUnits_X, workUnits_Y, workUnits_Z = noofThreads, 1, 1

	calRadius = time.time()
	for learning_rate_Index in range(0, NUM_LEARNING_RATES):
		print("learning_rate_Index", learning_rate_Index)
		thread_id = 0 + NUM_PARTITIONS * TOTAL_NUM_NEURONS * learning_rate_Index
		for j in range(0, NUM_PARTITIONS):
			for k in range(0, TOTAL_NUM_NEURONS):
				print("k ,thread_id", k, thread_id)
				print("thread_id", thread_id)
				feature_space_blockID = j
				rad_cal = ctypes.CDLL("/home/asim/SERIAL_CPU/Streaming_OPENCL_Classification/calculateradius.so")
				rad_cal.calculate_radius.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
																ctypes.c_int, ctypes.c_int, ctypes.c_int,
													 np.ctypeslib.ndpointer(dtype=np.int32,
																			flags="C_CONTIGUOUS"),
													 np.ctypeslib.ndpointer(dtype=np.int32,
																			flags="C_CONTIGUOUS"),
													 np.ctypeslib.ndpointer(dtype=np.int32,
																			flags="C_CONTIGUOUS"),
													 np.ctypeslib.ndpointer(dtype=np.int32,
																			flags="C_CONTIGUOUS"),
													 np.ctypeslib.ndpointer(dtype=np.int32,
																			flags="C_CONTIGUOUS"),

																np.ctypeslib.ndpointer(dtype=np.float32,
																					   flags="C_CONTIGUOUS"),
																np.ctypeslib.ndpointer(dtype=np.float32,
																					   flags="C_CONTIGUOUS"),
													 np.ctypeslib.ndpointer(dtype=np.float32,
																			flags="C_CONTIGUOUS"),
													 np.ctypeslib.ndpointer(dtype=np.float32,
																			flags="C_CONTIGUOUS"),
													 np.ctypeslib.ndpointer(dtype=np.float32,
																			flags="C_CONTIGUOUS"),



																np.ctypeslib.ndpointer(dtype=np.int32,
																					   flags="C_CONTIGUOUS"),

																ctypes.c_int, ctypes.c_int,ctypes.c_int]
		    	returnVale = rad_cal.calculate_radius(NUM_DATAPOINTS,
																TOTAL_NUM_NEURONS,
		                                                        total_num_features_all_featurespaces,
		                                                        TOTAL_NUM_FEATURES,
		                                                        NUM_PARTITIONS,
		                                                        NUM_NETS,
		                                                        NUM_CLASSES,
		                                                        NUM_FEATURES_PER_PARTITION,
		                                                        cumulative_no_features_featurespace,
		                                                        NUM_NEURONS_PER_NET,
		                                                        min_pos,
		                                                        active_centers_dominant_class,
		                                                        map_data,
		                                                        Input_Data,
		                                                        distance_map,
		                                                        min_array,
		                                                        radius_map,
		                                                        InputIndex_Per_Partition,
													            thread_id,
													            feature_space_blockID,
													            learning_rate_Index)
			#print("returnVale", returnVale)
			thread_id = thread_id + 1
	print("\nTime for calculate radius phase 2 : {0:.5f} secs".format(time.time()- calRadius))
	return map_data, active_centers_dominant_class, active_centers_total_count, active_centers_per_class, active_dominant_class_count, radius_map




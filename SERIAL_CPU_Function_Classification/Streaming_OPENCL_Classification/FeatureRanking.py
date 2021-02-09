def FeatureRanking():
    #print("Distances: "+str(overall_distances))
	
	context_django['total_training_time'] = context_django['total_training_time'] + (time.time() - active_centers_time)


	##print("Unsorted Distances:")
	i = 0
	j = 0
	for x in overall_distances:
		##print("Feature:"+str(i)+" Distance: "+str(x)+" Hyper: "+str(j))
		i = i + 1
		if i == tot_components:
			j = j + 1
			i = 0
	
	cust_type = np.dtype([('feature', np.int32),('dist', np.float32),('hyper',np.int32)])

	tot_components_per_iter = int(tot_components/number_of_iterations)
	##print("Testing"+ str(tot_components_per_iter))
	my_objects = np.empty(((no_hyper*number_of_iterations*no_of_classes, tot_components_per_iter)), dtype = cust_type)
	i = 0
	j = 0
	k = 0

	##print(my_objects.shape)
	##print("\n\n\nUnsorted:")
	for x1 in overall_distances:
		##print("Feature:"+str(new_input_column_names[i])+" "+str(i)+"\tDistance: "+str(x1)+"\tHyper: "+str(j))
		my_objects[j][i] = (i, x1, j)
		i = i + 1
		k = k + 1
		if i == tot_components_per_iter:
			j = j + 1
			i = 0
			##print("\n")
	##print("Unsorted:"+str(my_objects))
	#my_arr = np.empty(5, dtype = cust_type)
	my_object = np.array(my_objects, dtype = cust_type)

	def getKey(elem):
		return elem[1]

	sorted_obj = []

	
	i=0
	for x in range(0, no_hyper*number_of_iterations*no_of_classes):
		#(sorted_obj_hyp,), evt = sort(pycl_array.to_device(queue, my_object[x]), key_bits = 64, queue = queue)
		sorted_obj_hyp = sorted(my_object[x], key = getKey)
		sorted_obj.append(sorted_obj_hyp)	
		##print("Test 2 \n:"+str(i) + str(sorted_obj_hyp))
		i=i+1
	##print("Sorted using pyopencl:"+str(sorted_obj))

	new_input_column_names_iter = []
	for x in range(0,no_hyper*no_of_classes):
		new_input_column_names_iter = new_input_column_names_iter + (new_input_column_names)

	##print("Test1")
	##print(new_input_column_names_iter)
	#Generation of 'agg_coulmn_name' array that can be used too map agg results
	agg_coulmn_name = []
	j=0
	for x in input_column_names:
		agg_coulmn_name.append(input_column_names[j])
		j=j+1
	##print(agg_coulmn_name)
	##print("\n\n\nSorted:")
	k=0
	sum11 = []
	all_results = []
	all_results_names=[]
	baseindex = 0
	for x in range(0,no_of_classes):
		sum11 = []
		for r in range(0,tot_components_per_iter):
			sum11.append(0)

		##print(len(sum11))
		i = 0
		part_num = 0
		j=0
	
		for x in range(0, no_hyper*number_of_iterations):
			sorted_obj[baseindex + x].reverse()
			for x1 in sorted_obj[baseindex + x]:
				temp=x*tot_components_per_iter + x1['feature']
				j=0
				for y in input_column_names:
					if(agg_coulmn_name[j]==new_input_column_names_iter[(baseindex + x)*tot_components_per_iter + x1['feature']]):
						sum11[j] = sum11[j] + float(x1['dist'])
					j=j+1
				##print("Feature: "+str(new_input_column_names_iter[(baseindex + x)*tot_components_per_iter + x1['feature']])+"\tDistance: "+str(float(x1['dist']))+ "  Iteration: "+str(int((baseindex + x)/no_hyper)) +" ")
				i = i + 1
			i = 0
			#print("\n")

		sum11 = [x / (no_hyper * number_of_iterations) for x in sum11]
		all_results.append(sum11)
		baseindex = baseindex + (no_hyper * number_of_iterations)
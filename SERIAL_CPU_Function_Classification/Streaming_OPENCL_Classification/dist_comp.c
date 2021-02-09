#include<math.h>
void phase1_distanceComputation(
	int TOTAL_NUM_FEATURES,
	int No_Partitions,
	int TOTAL_NUM_NEURONS,
	int NUM_LEARNING_RATES,
	int NUM_NETS,
	int NUM_CLASSES,
	float *map_data,
	float *overall_distances,
	int *NUM_NEURONS_PER_NET,
	int *NUM_FEATURES_PER_PARTITION,
	int *active_centers,
        int tid,
        int leraning_rate_index
)
{

	// Each feature is a thread as per the shuffled order. NOT from original data order. 
	//int tid = get_global_id(0); // which feature
        //printf("\n tid %d",tid);
        //printf("\n leraning_rate_index %d", leraning_rate_index);     
	
	int partition_number = 0; // Calculate which partition this feature is in. 
	int part_offset = 0;
       
	for(int i = 0; i < No_Partitions; ++i)
	{
		if(tid < part_offset + NUM_FEATURES_PER_PARTITION[i])
			break;
		partition_number ++;
		part_offset += NUM_FEATURES_PER_PARTITION[i];
	}

	int base_map_position = (leraning_rate_index) * TOTAL_NUM_FEATURES * TOTAL_NUM_NEURONS; // (50 * 3 ) + (50 * 4) = (50 * 7)
	int feature_offset = 0;


	for(int i =0; i < partition_number; i++)
	{
		base_map_position += NUM_FEATURES_PER_PARTITION[i]*TOTAL_NUM_NEURONS;  
		feature_offset += NUM_FEATURES_PER_PARTITION[i];
	}	

	int neuronA = base_map_position;
	// float avg_dist = 0;	

	//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
 
        

	for(int class_no = 0; class_no < NUM_CLASSES ; ++class_no)
	{      
                //printf("\n class %d",class_no); 
                  int mybasedist = 0;
		  float din = 0;
		  float dout = 0;
		  int din_counter = 0;
		  int dout_counter = 0;		
		  int overall_counter = 0;	
		//float flag=0;
                  //float dino=0.5;
                //float din_tot=0;
                //float dout_tot=0;
               
            for(int nets = 0; nets< NUM_NETS; ++nets)
                {  
         
                   
                     
                   int net_offset=0+NUM_NEURONS_PER_NET[nets-1];
                       //printf("\noffset %d",net_offset);
            
		for(int j = 0; j< NUM_NEURONS_PER_NET[nets]; ++j)
                
		{
                      //printf("NUM_NEURONS %d ", NUM_NEURONS_PER_NET[nets]);
			for(int k = 0; k <NUM_NEURONS_PER_NET[nets]; ++k)
			{
                          
                           //printf("\nindex %f",((leraning_rate_index) * NUM_NEURONS_PER_NET[nets] * No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + j+net_offset));
if((active_centers[(leraning_rate_index) * NUM_NEURONS_PER_NET[nets] * No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + j+net_offset] == class_no) && (active_centers[(leraning_rate_index) * NUM_NEURONS_PER_NET[nets]* No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + k+net_offset] == class_no) && (active_centers[(leraning_rate_index) * NUM_NEURONS_PER_NET[nets] * No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + k+net_offset] != -1) && (active_centers[(leraning_rate_index) * NUM_NEURONS_PER_NET[nets] * No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + j+net_offset] != -1) && (j != k))
				{
					din += fabs(map_data[neuronA + (mybasedist + j) * NUM_FEATURES_PER_PARTITION[partition_number] + (tid-feature_offset)] - map_data[neuronA + (mybasedist + k) * NUM_FEATURES_PER_PARTITION[partition_number] + (tid-feature_offset)]);
					din_counter ++;
					overall_counter ++;
                               //printf("\ndIn: %f dincounter %d ",din,din_counter);
				}
					
				else if((active_centers[(leraning_rate_index) * NUM_NEURONS_PER_NET[nets] * No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + j+net_offset] == class_no) && (active_centers[(leraning_rate_index) * NUM_NEURONS_PER_NET[nets] * No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + k+net_offset] != class_no) && (active_centers[(leraning_rate_index) * NUM_NEURONS_PER_NET[nets] * No_Partitions + NUM_NEURONS_PER_NET[nets] * partition_number + mybasedist + k+net_offset] != -1))
				{

					dout += fabs(map_data[neuronA + (mybasedist + j) * NUM_FEATURES_PER_PARTITION[partition_number] + (tid-feature_offset)] - map_data[neuronA + (mybasedist + k) * NUM_FEATURES_PER_PARTITION[partition_number] + (tid-feature_offset)]);
					dout_counter ++;
					overall_counter ++;
                                //printf("\ndout: %f doutcounter %d ",dout,dout_counter);
				}
			}
		}
		// if(leraning_rate_index == 0 && blockIdx.x < 10)
		//printf("\nthreaddIn: %f ",din);
               //float din_tot=din_tot+din;
               //float dout_tot=dout_tot+dout;
               
               
           }
               
		float avg_dOut = (float)(dout) / (float)(dout_counter);
                //printf("\navg_dOut %f",avg_dOut);
		float avg_dIn = (float)(din) / (float)(din_counter);
                //printf("\ndin_counter %f",din_counter);
                if (avg_dIn==0)
                 {
                   avg_dIn=0.001+avg_dIn;
                 }
		float ratioDinDout = avg_dOut/avg_dIn;
                //printf("\nratioDinDout %f",ratioDinDout);
		//float ratioDinDout=dino;
		if(dout_counter == 0 || din_counter==0)
		{
			ratioDinDout = 0.001*tid;
		}
                //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
                //printf("\n class %d",class_no);
               //printf("\n class no %d  leraning_rate_index %d total_num_features %d threadid %d ratioDinDout %f",class_no, leraning_rate_index, TOTAL_NUM_FEATURES, tid,ratioDinDout);
                 //printf("\n index %d tid %d",( class_no *NUM_LEARNING_RATES * TOTAL_NUM_FEATURES + 0 * TOTAL_NUM_FEATURES + tid), tid);
		overall_distances[class_no  * NUM_LEARNING_RATES *TOTAL_NUM_FEATURES + leraning_rate_index * TOTAL_NUM_FEATURES + tid] = ratioDinDout;
                //printf("\nratioDinDout %f",ratioDinDout);
               //printf("\n class %d thread_id %d overall_distances %f",class_no,tid,ratioDinDout); 
               
               
	}
      
      
}

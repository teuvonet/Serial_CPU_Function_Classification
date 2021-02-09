/*
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.

*/


#include<math.h>
void calculate_radius(
    int NUM_DATAPOINTS,
    int TOTAL_NUM_NEURONS,
    int total_num_features_all_featurespaces,
    int TOTAL_NUM_FEATURES,
    int NUM_PARTITIONS,
    int NUM_NETS,
    int NUM_CLASSES,
    int *NUM_FEATURES_PER_PARTITION,
    int *cumulative_no_features_featurespace,
    int *neurons_per_nets,
    int *min_pos,
    int *active_centers,
    float *map_data,
    float *Input_Data,
    float *distance_map,
    float *min_dist,
    float *radius_map,
    int *InputIndex_Per_Partition,
    int thread_id,
    int feature_space_blockID,
    int learning_rate_Index
    )
{	
    // Finding the total number of threads in the program
    //const int block_skip = get_local_size(2)*get_local_size(1)*get_local_size(0);

    // Finding thread index using block and thread index
    //const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0);

    //Initial declaration
    //int feature_space_blockID = get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
    int no_of_features_featurespace = NUM_FEATURES_PER_PARTITION[feature_space_blockID]; //Get the Number of features that in each feature space
    //int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread


    //Find the corresponding Map Array Index
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces;
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID];
    map_start_index += no_of_features_featurespace * (thread_id % TOTAL_NUM_NEURONS);
    
    
    // Loop for each data point
    int datapoint = 0;
    for(; datapoint < NUM_DATAPOINTS; datapoint++)
    {
        
        //______________________________________________________________________________________________________________________________________________________
        // Step 1: Find Cumulative distance between neuron center and input data for each feature
        //------------------------------------------------------------------------------------------------------------------------------------------------------
        
        float sum=0;

        for (int feature = 0; feature < no_of_features_featurespace; feature++){
            sum += fabs(Input_Data[datapoint * TOTAL_NUM_FEATURES + feature] - map_data[map_start_index + feature]);
        }
        distance_map[thread_id] = sum;
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        //______________________________________________________________________________________________________________________________________________________
        //END OF STEP 1
        //----------------------------------------------------------------------------------------------------------------------------------------------------------



        //_____________________________________________________________________________________________________________________________________________________
        //STEP 3: Calculating minium distance Neuron and it's index in that Net
        //----------------------------------------------------------------------------------------------------------------------------------------------------

        if((NUM_PARTITIONS*TOTAL_NUM_NEURONS)%TOTAL_NUM_NEURONS == 0)
        {
            //Distance Map index adjusting
            int distance_map_Index = thread_id;
            int min_pos_startIndex = (thread_id/TOTAL_NUM_NEURONS)*NUM_NETS;
            int active_centers_Index = thread_id;
            int radius_map_index = thread_id;

            //Loop through each net
            int net = 0;
            for(; net < NUM_NETS; net++)
            {
                min_dist[min_pos_startIndex + net] = __INT_MAX__;
                
                //Loop through every neuron in that net
                for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                {
                    if(distance_map[distance_map_Index + neuron] < min_dist[min_pos_startIndex + net] && active_centers[active_centers_Index + neuron] != -1)
                    {
                        //Capture min dist array neuron and it's Index
                        min_dist[min_pos_startIndex + net] = distance_map[distance_map_Index + neuron];
                        min_pos[min_pos_startIndex + net] = neuron;
                    }
                }

                int winner = min_pos[min_pos_startIndex + net];
                float winnerDistance = min_dist[min_pos_startIndex + net];
                
                if(active_centers[active_centers_Index + winner] != -1)
                {
                    radius_map[radius_map_index + winner] = ( radius_map[radius_map_index + winner] > winnerDistance ) ? radius_map[radius_map_index + winner] : winnerDistance;
                }
                
                //Adjust distance map after every net
                distance_map_Index += neurons_per_nets[net];
                radius_map_index += neurons_per_nets[net];
            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);  
        
        //______________________________________________________________________________________________________________________________________________________
        //END OF STEP 2
        //----------------------------------------------------------------------------------------------------------------------------------------------------------
        
    }

}

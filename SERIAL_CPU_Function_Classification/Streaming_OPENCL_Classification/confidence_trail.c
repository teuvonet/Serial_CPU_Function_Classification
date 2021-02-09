
/*
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.
*/

#include<math.h>
void Confidence_Score(
    int NUM_DATAPOINTS_OVERALL,
    int TOTAL_NUM_FEATURES,
    int TOTAL_NUM_NEURONS,
    int total_num_features_all_featurespaces,
    int NUM_NETS,
    int NUM_PARTITIONS,
    int NUM_CLASSES,
    int *NUM_FEATURES_PER_PARTITION,
    int *active_centers_dominant_class,
    int *active_centers_total_count,
    int *active_dominant_count,
    int *active_centers_per_class,
    int *min_pos,
    int *prediction_per_part_array,
    int *cumulative_no_features_partition,
    float *Input_Data,
    float *map,
    float *distance_map,
    float *min_array,
    float *radius_map,
    float *confidence_array,
    int *KNN_Winners_ClassWise_Count,
    float *summa,
    float *red_map,
    int *dominant_count_first,
    int *dominant_count_second,
    int *dominant_count_third,
    float *confidence_phase_3,
    float *firstWinnerDis,
    float *secondWinnerDis,
    float *ThirdWinnerDis,
    int *total_count_first,
    int *total_count_second,
    int *total_count_third,
    int *maxconfidenceclass,
    float *confidence_winner,
    float *confidence_second,
    float *confidence_third,
    int *dominant_class_winner,
    int *dominant_class_second,
    int *dominant_class_third,
    int *winner_neuron,
    int *second_neuron,
    int *third_neuron,
    int thread_id,
    int Partition_Block_Id,
    int learning_rate_Index
   
)
{
    // Finding the total number of threads in the program
    //const int block_skip = blockDim.z*blockDim.y*blockDim.x;

    // Finding thread index using block and thread index
    //const int thread_id =get_global_id(0) + get_global_id(1)*get_global_size(0);
    //printf("\n thread_id %d",thread_id);
    //Gathering Initial values
    //int Partition_Block_Id = get_global_id(0)/TOTAL_NUM_NEURONS; //Partition Number
    //printf("\n Partition_Block_Id %d",Partition_Block_Id);
    int no_of_features_partition = NUM_FEATURES_PER_PARTITION[Partition_Block_Id]; //Get the Number of features that in each feature space
    //int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread
    //printf("\n thread_id %d",blockIdx.x);
    //Find the corresponding Map Array Index
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces;
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_partition[Partition_Block_Id];
    map_start_index += no_of_features_partition * (thread_id % TOTAL_NUM_NEURONS);
    //printf("\n thread_id %d threadIdx %d blockIdx %d map_start_index %d",thread_id,threadIdx.x,blockIdx.x,map_start_index);
    //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);




   for(int datapoint = 0; datapoint < NUM_DATAPOINTS_OVERALL; datapoint++)
   //for(int datapoint = 0; datapoint < 1; datapoint++)
   {

        //Looping thorugh each feature for every neuron and finding the cumulative sum between the neuron and Input data
        float sum = 0;
        
        for (int feature = 0; feature < no_of_features_partition ; feature++)
        {
            sum += fabs(Input_Data[datapoint * TOTAL_NUM_FEATURES  + feature] - map[map_start_index + feature]);
        }
        summa[datapoint]=sum;
        //printf("\n sum %f",summa[datapoint]);
        // printf("\n radius map %f", radius_map[datapoint]);
        //Index in the distance = thread_id 
        distance_map[thread_id] = sum;
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
       //red_map[datapoint]=radius_map[datapoint];

        if((NUM_PARTITIONS*TOTAL_NUM_NEURONS)%TOTAL_NUM_NEURONS == 0)
        {
            int min_pos_startIndex = learning_rate_Index * NUM_PARTITIONS + Partition_Block_Id ;
            min_array[min_pos_startIndex] =__INT_MAX__ ;
            
            //Distance Map index adjusting
            int distance_map_Index = thread_id;
            
            int active_centers_dominant_Index = thread_id;

            int secondWinner = -1;
            int thirdWinner = -1;
            float secondWinnerDist = __INT_MAX__;
            float ThirdWinnerDist = __INT_MAX__;
            float firstWinnerDist;
            //printf("\n min_pos_startIndex %d distance_map_Index %d  active_centers_dominant_Index %d",min_pos_startIndex,distance_map_Index,active_centers_dominant_Index  );
            for(int neuron = 0; neuron < TOTAL_NUM_NEURONS ; neuron++)
            {
              if(active_centers_dominant_class[active_centers_dominant_Index + neuron] != -1 && active_dominant_count[active_centers_dominant_Index + neuron]> NUM_DATAPOINTS_OVERALL*0.01)
                {
                if(distance_map[distance_map_Index + neuron] < min_array[min_pos_startIndex]  )
                {			        
                    min_array[min_pos_startIndex] = distance_map[distance_map_Index + neuron];
                    thirdWinner=secondWinner;
                    ThirdWinnerDist = secondWinnerDist;
                    secondWinner = min_pos[min_pos_startIndex]; 
                    secondWinnerDist = min_array[min_pos_startIndex];
                    min_pos[min_pos_startIndex] =  neuron;
                    
                    //if(thread_id==0){
                    //printf("\n %f", firstWinnerDist);}
                    //printf("\n winner %d",min_pos[min_pos_startIndex]);
                }
                else if(distance_map[distance_map_Index + neuron] > min_array[min_pos_startIndex] && distance_map[distance_map_Index + neuron] < secondWinnerDist){
                    
                    //if(thread_id==0){
                    //printf("\n %f", secondWinnerDist);}
                    thirdWinner=secondWinner;
                    ThirdWinnerDist = secondWinnerDist;
                    secondWinner = neuron;
                    secondWinnerDist = distance_map[distance_map_Index + neuron];
                    //printf("\n secondWinner %d",secondWinner);
                }
                else if(distance_map[distance_map_Index + neuron] > secondWinnerDist && distance_map[distance_map_Index + neuron] < ThirdWinnerDist){
                    
                    //printf("\n ThirdWinnerDist %f,thraes_id %d", ThirdWinnerDist,thread_id);
                    ThirdWinnerDist = distance_map[distance_map_Index + neuron]; 
                    thirdWinner = neuron;
                    //printf("\n thirdWinner %d",thirdWinner);
                }
             
              }  

            }
           
           
            //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
           
            //printf("\n winner %d second %d third %d",min_pos[min_pos_startIndex],secondWinner,thirdWinner );
            int winner = min_pos[min_pos_startIndex];
            int active_centers_per_class_Index = thread_id * NUM_CLASSES;
            int KNN_Winners_ClassWise_Count_Index = learning_rate_Index * NUM_PARTITIONS + Partition_Block_Id;
            int all_winner_count = 0;
            //printf("\n winner %d active_centers_per_class_Index %d KNN_Winners_ClassWise_Count_Index %d",winner,active_centers_per_class_Index,KNN_Winners_ClassWise_Count_Index);
            //printf("\n winner neuron %d",winner);
            for(int clas = 0; clas < NUM_CLASSES; clas++){
                int currentClassTotalCount = 0;
                currentClassTotalCount += active_centers_per_class[active_centers_per_class_Index + winner * NUM_CLASSES + clas];
                //printf("\n winner class count: %d clas: %d",currentClassTotalCount, clas);
                
                currentClassTotalCount += active_centers_per_class[active_centers_per_class_Index + secondWinner * NUM_CLASSES + clas];
                
                currentClassTotalCount += active_centers_per_class[active_centers_per_class_Index + thirdWinner * NUM_CLASSES + clas];
                
                //printf("\n currentClassTotalCount %d",currentClassTotalCount);
                KNN_Winners_ClassWise_Count[KNN_Winners_ClassWise_Count_Index + clas] = currentClassTotalCount;
                all_winner_count += currentClassTotalCount;
            }

            float ratio_fir_sec = ((float)distance_map[distance_map_Index + secondWinner])/((float)distance_map[distance_map_Index + winner]); 
            float rato_fir_third = ((float)distance_map[distance_map_Index + thirdWinner])/((float)distance_map[distance_map_Index + winner]);
       

            int KNN_Winners_ClassWise_Index =thread_id; 
            int maxconfidence=confidence_phase_3[KNN_Winners_ClassWise_Index+winner];
            int maxconfidenceclass=active_centers_dominant_class[KNN_Winners_ClassWise_Index+winner];
            dominant_count_first[datapoint]=active_dominant_count[active_centers_dominant_Index + winner];
            int maxconfidence_second =confidence_phase_3[KNN_Winners_ClassWise_Index+secondWinner];
            int maxconfidenceclass_second=active_centers_dominant_class[KNN_Winners_ClassWise_Index+secondWinner];
            dominant_count_second[datapoint]=active_dominant_count[active_centers_dominant_Index + secondWinner];
            int maxconfidence_third =confidence_phase_3[KNN_Winners_ClassWise_Index+thirdWinner];
            int maxconfidenceclass_third=active_centers_dominant_class[KNN_Winners_ClassWise_Index+thirdWinner];
            dominant_count_third[datapoint]=active_dominant_count[active_centers_dominant_Index + thirdWinner];
            if(maxconfidence_second> maxconfidence && ratio_fir_sec < 2.0 )
            {
            maxconfidence=maxconfidence_second;
            maxconfidenceclass=maxconfidenceclass_second;
            }
            if (maxconfidence_third> maxconfidence && rato_fir_third < 2.0 )
            {

            maxconfidence=maxconfidence_third;
            maxconfidenceclass=maxconfidenceclass_third;

            }
           
            int maxCountClass = 0;
            int maxCount = 0;
            int temp = 0;
            int temp_Index = thread_id;
            temp += active_centers_total_count[temp_Index + winner];
           
            temp += active_centers_total_count[temp_Index + secondWinner];
            
            temp += active_centers_total_count[temp_Index + thirdWinner];
            
           
            for(int clas = 0; clas < NUM_CLASSES; clas++){
               
                if(fabs(radius_map[datapoint]-(distance_map[datapoint]))>0.0)
                  { 
                    //printf("\n radius_map %f",radius_map[datapoint]);
                    maxCountClass = clas;
                    //printf("\n maxcount inside the radius class %d datapoint %d thread_id %d",maxCountClass,datapoint,thread_id);
                   }
                              
                else if(KNN_Winners_ClassWise_Count[KNN_Winners_ClassWise_Count_Index + clas] > maxCount)
                {
                    maxCount =  KNN_Winners_ClassWise_Count[KNN_Winners_ClassWise_Count_Index + clas];
                    maxCountClass = clas;
                    //printf("\n knn maxout %ddatapoint %d thread_id %d",maxCountClass,datapoint,thread_id);
                    
                    //printf("\n winner based on kNN manasa method %d",maxCountClass);
                }
                
            }
            //if(i==1)
            //{
            //printf("\n it is inside");
             //}
            
            //printf("\n winner neuron %d", Winner);
            //printf("\n maxout otside the loop manasa method %d class %d",maxCountClass,maxCount);
            //printf("\n active centers dominant count %d", active_centers_dominant);

            float confidence = (float)maxCount/ (float)all_winner_count;
             total_count_first[datapoint]=active_centers_total_count[active_centers_dominant_Index + winner];
             total_count_second[datapoint]=active_centers_total_count[active_centers_dominant_Index + secondWinner];
             total_count_third[datapoint]=active_centers_total_count[active_centers_dominant_Index + thirdWinner];

            // int winner_total_ClassCount = active_centers_total_count[active_centers_dominant_Index + winner];
            // int winner_class_count = active_dominant_count[active_centers_dominant_Index + winner];
            // float winner_confidence = (float)winner_class_count/ (float)winner_total_ClassCount;
            confidence_winner[datapoint]=(float)dominant_count_first[datapoint]/((float)total_count_first[datapoint]);
            confidence_second[datapoint]=(float)dominant_count_second[datapoint]/((float)total_count_second[datapoint]);
            confidence_third[datapoint]=(float)dominant_count_third[datapoint]/((float)total_count_third[datapoint]);
            //printf("\n %f",confidence_winner[datapoint]);
            
            firstWinnerDis[datapoint] = distance_map[distance_map_Index + winner];
            secondWinnerDis[datapoint] = distance_map[distance_map_Index + secondWinner];
            ThirdWinnerDis[datapoint] = distance_map[distance_map_Index + thirdWinner];

            dominant_class_winner[datapoint]=active_centers_dominant_class[active_centers_dominant_Index + winner];
            dominant_class_second[datapoint]=active_centers_dominant_class[active_centers_dominant_Index + secondWinner];
            dominant_class_third[datapoint]=active_centers_dominant_class[active_centers_dominant_Index + thirdWinner];

            winner_neuron[datapoint] = winner;
            second_neuron[datapoint] = secondWinner;
            third_neuron[datapoint] = thirdWinner;
            // printf("\n datapoint %d firstwinner %d secondwineer %d third winner %d maxconfidence class %d thread_id %d ",datapoint,dominant_class_winner[datapoint],dominant_class_second[datapoint],dominant_class_third[datapoint],maxconfidenceclass, thread_id );
            int Confidence_Prediction_Array_StartIndex = learning_rate_Index * NUM_PARTITIONS * NUM_DATAPOINTS_OVERALL;
            
            Confidence_Prediction_Array_StartIndex += datapoint + (Partition_Block_Id * NUM_DATAPOINTS_OVERALL);

            //if(Partition_Block_Id == 20 && datapoint <= 4 && learning_rate_Index == 0){
                //printf("\n datapoint %d neuron %d all_winner_count :%d  fromactiv :%d maxCount %d maxCountClass %d", datapoint, thread_id,  all_winner_count, temp , maxCount, maxCountClass);
            //}

            confidence_array[Confidence_Prediction_Array_StartIndex] = maxconfidence;
            prediction_per_part_array[Confidence_Prediction_Array_StartIndex] = maxconfidenceclass;
        }
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

import os
import glob
import pandas as pd
os.chdir("/home/asim/OPENCL_CLASSIFICAT_learning_3/Streaming_OPENCL_Classification/Datasets/Combined_Pima")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ],sort='False')
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

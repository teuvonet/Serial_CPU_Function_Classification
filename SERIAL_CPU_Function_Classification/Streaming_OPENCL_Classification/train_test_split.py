from numpy.random import RandomState
import pandas as pd

df = pd.read_csv('/home/asim/OPENCL_CLASSIFICAT_learning_3/Streaming_OPENCL_Classification/Datasets/Combined_Pima/combined_csv.csv')
rng = RandomState()

train = df.sample(frac=0.9, random_state=rng)
test = df.loc[~df.index.isin(train.index)]
train.to_csv('/home/asim/OPENCL_CLASSIFICAT_learning_3/Streaming_OPENCL_Classification/Datasets/Combined_Pima/train.csv')
test.to_csv('/home/asim/OPENCL_CLASSIFICAT_learning_3/Streaming_OPENCL_Classification/Datasets/Combined_Pima/test.csv')

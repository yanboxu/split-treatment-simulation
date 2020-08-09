import pandas as pd 
import numpy as np 

file_name_list=[ "Obs_EndUserFeature_pruned",  "Obs_O365UserFeature_pruned"]

for fname in file_name_list:
	# Loading data from no columns csv files; hence header=None set
	data_frame=  pd.read_csv(fname+".csv", header=None)
	print(data_frame.shape)
	data_types= data_frame.dtypes

	data_size= data_frame.shape[0]
	random_col= np.random.rand(data_size)
	random_col[random_col<0.5]=0
	random_col[random_col>=0.5]= 1
	print( np.unique(random_col, return_counts=True) )
	random_col= random_col.astype(int)
	# Treatment Column Updated with random treatment assignments
	data_frame[2]= random_col

	# Save the file: CAUTION: index=False to avoid adding the extra index column
	data_frame.to_csv( fname+"_refute.csv", index=False, header=None )

	# Test
	data_frame=  pd.read_csv(fname+"_refute.csv", header=None)
	print(data_frame.shape)
	# Check to see if data types are changed while saving
	if not data_types.equals(data_frame.dtypes):
		print("Error: Data Types of columns don't match")


# temp= pd.read_csv("temp.csv", header=None)
# print(temp.shape, temp.dtypes)

# data_size= temp.shape[0]
# random_col= np.random.rand(data_size)
# random_col[random_col<0.5]= 0
# random_col[random_col>=0.5]= 1
# random_col= random_col.astype(int)
# temp[2]= random_col

# temp.to_csv("temp_refute.csv", index=False, header=None)

# temp= pd.read_csv("temp_refute.csv", header=None)
# print(temp.shape, temp.dtypes)

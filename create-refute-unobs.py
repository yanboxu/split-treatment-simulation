import pandas as pd 
import numpy as np 
from scipy.stats.stats import pearsonr 

file_name_list=[  "Obs_Feature25_pruned", "Obs_Feature106_pruned"]

for fname in file_name_list:
	# Loading data from no columns csv files; hence header=None set
	data_frame=  pd.read_csv(fname+".csv", header=None)
	print( 'Dataset Size', data_frame.shape)
	data_types= data_frame.dtypes

	# Getting the data corresponding to the treatment and the outcome column
	yname= 1 #'SustainedUseDays'
	tname= 2 #'Treated'
	treatment= data_frame[tname] 
	outcome= data_frame[yname]
	total_size= data_frame.shape[0]
	unobs_confounder_stats=np.zeros((2,2))

	if fname == 'Obs_Feature106_pruned':
		alpha= 1000.0
		eps= 1700*alpha
	else:		
		alpha= 1000.0
		eps= 600*alpha

	#Compute the posterior distirbution of the unobs confounder 
	for tcase in [0,1]:
		sample_outcome= outcome[treatment==tcase].to_numpy()
		sample_size= outcome.shape[0]

		# Posterior Mean
		unobs_confounder_stats[tcase,0]= ( alpha + tcase + sample_size*np.mean(sample_outcome))/(sample_size+1)
		# Posterior Vairance
		unobs_confounder_stats[tcase,1]= eps/(sample_size+1)
	print('Posterior Distirbution: ', unobs_confounder_stats)

	# Sample from the unobs confounder posterior
	unobs_confounder= np.zeros((total_size))
	mu= np.zeros((total_size))
	sigma= np.zeros((total_size))
	z= np.random.normal(0,1,total_size)
	for tcase in [0,1]:
		mu[treatment==tcase]= unobs_confounder_stats[tcase,0]
		sigma[treatment==tcase]= np.sqrt( unobs_confounder_stats[tcase,1] ) 

	unobs_confounder= mu + sigma*z

	# Compute the correlation of U with T,Y
	print( 'U Shape', unobs_confounder.shape )
	print( 'U Val', unobs_confounder[ treatment == 0 ][:20],  unobs_confounder[ treatment == 1 ][:20] )
	corr_treat= pearsonr( unobs_confounder, treatment.to_numpy() )
	corr_out= pearsonr( unobs_confounder, outcome.to_numpy() )
	print( 'Correlation Treatment: ', corr_treat )
	print( 'Correlation_Outcome:' , corr_out )

	# Add the new column to the datatframe
	unobs_confounder= np.reshape( unobs_confounder, (unobs_confounder.shape[0]) )
	data_frame[-1]= unobs_confounder

	# Save the file: CAUTION: index=False to avoid adding the extra index column
	data_frame.to_csv( fname+"_refute.csv", index=False, header=None )

	# Test
	data_frame=  pd.read_csv(fname+"_refute.csv", header=None)
	print(data_frame.shape)

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

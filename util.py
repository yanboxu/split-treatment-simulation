import numpy as np
import random

import os
import pickle
import sys
import timeit

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_spd_matrix
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.stats import pearsonr 
from sklearn import svm

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import FormatStrFormatter

from matplotlib.pyplot import imshow

import pickle

def sim_Data(xDim, nSim, p_AgivenZ, p_AgivenNotZ):
    #** Follow the graph to generate Y
    # Y = X * beta + Z * CATE

    # confounders
    x_Sigma = make_spd_matrix(xDim)
    X = np.random.multivariate_normal(np.zeros(xDim),x_Sigma,size=nSim)
    beta = np.random.choice(5, xDim, replace=True, p =[.3, .25, .2, .15, .1])

    # ground truth CATE
    CATE = np.array([10,20,30,40])
    p_G = np.exp(X[:,1])/(1+np.exp(X[:,1]))
    Group = np.zeros(nSim)
    Group[p_G < 0.75] = 1
    Group[p_G < 0.5] = 2
    Group[p_G < 0.25] = 3
    Group = Group.astype(int)

    # Z encourage A, and split-treatment criterion p_A_given_Z_X > p_A_given_notZ_X
    p_A_given_Z = [p_AgivenZ for x in X] #if x[0] > 0 else p_AgivenZ-0.1 
    p_A_given_notZ = [p_AgivenNotZ for x in X]
    Compliance = np.array(p_A_given_Z) - np.array(p_A_given_notZ)
    print('Avg compliance:', np.mean(Compliance))

    # randomized treatment
    Z = np.random.choice(2, nSim)

    A = [np.random.choice(2, 1, p = [1-p_A_given_Z[i], p_A_given_Z[i]]) 
         if Z[i] == 1 else np.random.choice(2, 1, p = [1-p_A_given_notZ[i], p_A_given_notZ[i]])
        for i in range(nSim)]

    # ground truth two-arm potential outcomes
    Y_0 = np.random.normal(np.sum(X * beta,1),1)
    #Y_1 = Y_0 + np.random.normal(CATE[Group],1)
    Y_1 = Y_0 + CATE[Group]/(Compliance * 0.5)
    
    Y = [Y_0[i] if A[i] == 0 else Y_1[i] for i in range(nSim)]

    Z = np.array(Z).ravel()
    A = np.array(A).ravel()
    Y = np.array(Y).ravel()
    
#     print('Z==1:',sum(Z), 'A==Z:',sum(A*Z))
    
    # return full observed data
    return X, Y, A, nSim, Group, Y_0, Y_1, Z, A

def sim_Unobs_Data(xDim, nSim, p_AgivenZ, p_AgivenNotZ):
    #** Follow the graph to generate Y
    # Y = X * beta + U * gamma + Z * CATE

    #** Add a U to X, A and Y
    # confounders
    x_Sigma = make_spd_matrix(xDim)
    X = np.random.multivariate_normal(np.zeros(xDim),x_Sigma,size=nSim)
    beta = np.random.choice(5, xDim, replace=True, p =[.3, .25, .2, .15, .1])
    
    U = np.random.normal(0.5 * np.ones(nSim),1)
    gamma = 2

    # ground truth CATE
    CATE = np.array([1,2,3,4])
    p_G = np.exp(X[:,1])/(1+np.exp(X[:,1]))
    Group = np.zeros(nSim)
    Group[p_G < 0.75] = 1
    Group[p_G < 0.5] = 2
    Group[p_G < 0.25] = 3
    Group = Group.astype(int)


    # Z encourage A, and split-treatment criterion p_A_given_Z_X > p_A_given_notZ_X
    p_A_given_Z = [p_AgivenZ for x in X] #if x[0] > 0 else p_AgivenZ-0.1 
    p_A_given_notZ = [p_AgivenNotZ for x in X]
    Compliance = np.array(p_A_given_Z) - np.array(p_A_given_notZ)
    print('Avg compliance:', np.mean(Compliance))

    # randomized treatment
    Z = np.random.choice(2, nSim)

    A = [np.random.choice(2, 1, p = [1-p_A_given_Z[i], p_A_given_Z[i]]) 
         if Z[i] == 1 else np.random.choice(2, 1, p = [1-p_A_given_notZ[i], p_A_given_notZ[i]])
        for i in range(nSim)]

    # ground truth two-arm potential outcomes
    Y_0 = np.random.normal(np.sum(X * beta,1) + U * gamma,1)
    #Y_1 = Y_0 + np.random.normal(CATE[Group],1)
    Y_1 = Y_0 + CATE[Group]/(Compliance * 0.5)
    
    Y = [Y_0[i] if A[i] == 0 else Y_1[i] for i in range(nSim)]

    Z = np.array(Z).ravel()
    A = np.array(A).ravel()
    Y = np.array(Y).ravel()
    
#     print('Z==1:',sum(Z), 'A==Z:',sum(A*Z))
    
    # return full observed data
    return X, Y, A, nSim, Group, Y_0, Y_1, Z, A

def evaluation(Yhat_0, Yhat_1, Group_data, Y_0_data, Y_1_data, A, Z):

    print('Est. Y_0 RMSE:', np.sqrt(np.mean(Yhat_0 - Y_0_data) ** 2))
    print('Est. Y_1 RMSE:', np.sqrt(np.mean(Yhat_1 - Y_1_data) ** 2))
    
    est_ite_sort_order = np.argsort(Yhat_1 - Yhat_0)
    recovered_rank = np.zeros(len(Z))
    n_start = 0
    for i in range(len(np.unique(Group_data))):
        n_i = np.sum(Group_data == i)
        recovered_rank[est_ite_sort_order[n_start: (n_start + n_i)]] = i+1
        n_start += n_i
        
    ground_truth_rank = Group_data + 1
    rmse = np.sqrt(np.mean((recovered_rank - ground_truth_rank)**2))
    
    A_Z_matched = sum(A == Z)/len(Z)
    print('{:.1f}% A matched Z, RMSE: {:.2f}'.format(A_Z_matched * 100., rmse))
    #print('Ground truth CATE:', CATE)
    
    ground_truth_sort_order = np.argsort(ground_truth_rank)
    
    fig = plt.figure(figsize=(10,6))

    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(ground_truth_rank[ground_truth_sort_order])
    ax1.set_title('Ground truth CATE')

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(recovered_rank[ground_truth_sort_order],lw=0.5)
    ax2.set_title('Recovered Rank (RMSE = {:.2f})'.format(rmse))

    fig.suptitle('{:.1f}% A matched Z'.format(A_Z_matched * 100.))
    
    return A_Z_matched, rmse



def fit_IPTW_LR(X_data, Y_data, A_data, nObs):
    # 1. Fit propensity
    clf = LogisticRegression().fit(X_data, A_data)
    propensity = clf.predict_proba(X_data)[:,1]
    p_A = np.sum(A_data)/len(A_data)
    iptw = [p_A/propensity[i] if A_data[i] == 1 else (1-p_A)/(1-propensity[i]) for i in range(nObs)]

    print('p_A: {}, IPTW: {:.2f} +/- {:.2f}'.format(p_A, np.mean(iptw), np.std(iptw)))
    
    # 2. Fit weighted LR
    X_A_data = np.array([X_data[i] * A_data[i] for i in range(nObs)])
    model = LinearRegression(fit_intercept=False).fit(np.concatenate((X_data,X_A_data, A_data.reshape(-1,1)),1),Y_data,sample_weight=iptw)
    Yhat_0 = model.predict(np.concatenate((X_data, np.zeros(X_data.shape), np.zeros((nObs,1))),1))
    Yhat_1 = model.predict(np.concatenate((X_data, X_data, np.ones((nObs,1))),1))

    return Yhat_0, Yhat_1

def fit_IPTW_RF(X_data, Y_data, A_data, nObs):
    # 1. Fit propensity
    clf = LogisticRegression().fit(X_data, A_data)
    propensity = clf.predict_proba(X_data)[:,1]
    p_A = np.sum(A_data)/len(A_data)
    iptw = [p_A/propensity[i] if A_data[i] == 1 else (1-p_A)/(1-propensity[i]) for i in range(nObs)]

    print('p_A: {}, IPTW: {:.2f} +/- {:.2f}'.format(p_A, np.mean(iptw), np.std(iptw)))
    
    # 2. Fit weighted RF
    X_A_data = np.array([X_data[i] * A_data[i] for i in range(nObs)])
    model = RandomForestRegressor().fit(np.concatenate((X_data, A_data.reshape(-1,1)),1),Y_data,sample_weight=iptw)
    Yhat_0 = model.predict(np.concatenate((X_data, np.zeros((nObs,1))),1))
    Yhat_1 = model.predict(np.concatenate((X_data, np.ones((nObs,1))),1))

    return Yhat_0, Yhat_1

def fit_IPTW_SVM(X_data, Y_data, A_data, nObs):
    # 1. Fit propensity
    clf = LogisticRegression().fit(X_data, A_data)
    propensity = clf.predict_proba(X_data)[:,1]
    p_A = np.sum(A_data)/len(A_data)
    iptw = [p_A/propensity[i] if A_data[i] == 1 else (1-p_A)/(1-propensity[i]) for i in range(nObs)]

    print('p_A: {}, IPTW: {:.2f} +/- {:.2f}'.format(p_A, np.mean(iptw), np.std(iptw)))
    
    # 2. Fit weighted RF
    X_A_data = np.array([X_data[i] * A_data[i] for i in range(nObs)])
    model = svm.SVR(kernel='linear').fit(np.concatenate((X_data, X_A_data, A_data.reshape(-1,1)),1),Y_data,sample_weight=iptw)
    Yhat_0 = model.predict(np.concatenate((X_data,np.zeros(X_data.shape), np.zeros((nObs,1))),1))
    Yhat_1 = model.predict(np.concatenate((X_data, X_data, np.ones((nObs,1))),1))

    return Yhat_0, Yhat_1


def fit_LR(X_data, Y_data, A_data, nObs):
    # Fit LR
    X_A_data = np.array([X_data[i] * A_data[i] for i in range(nObs)])
    model = LinearRegression(fit_intercept=False).fit(np.concatenate((X_data,X_A_data, A_data.reshape(-1,1)),1),Y_data)
    Yhat_0 = model.predict(np.concatenate((X_data, np.zeros(X_data.shape), np.zeros((nObs,1))),1))
    Yhat_1 = model.predict(np.concatenate((X_data,X_data, np.ones((nObs,1))),1))

    return Yhat_0, Yhat_1


def placebo_confounder(X_data, A_data, Y_data):

    treatment= A_data 
    outcome= Y_data
    data_size= X_data.shape[0]
    random_col= np.random.rand(data_size)
    random_col[random_col<0.5]=0
    random_col[random_col>=0.5]= 1
    print( np.unique(random_col, return_counts=True) )
    A_data_placebo= random_col.astype(int)
    return A_data_placebo


def bayes_unobs_confounder(X_data, A_data, Y_data, alpha, eps):

    treatment= A_data 
    outcome= Y_data
    total_size= X_data.shape[0]
    unobs_confounder_stats=np.zeros((2,2))
    
    #Compute the posterior distirbution of the unobs confounder 
    for tcase in [0,1]:
        sample_outcome= outcome[treatment==tcase]
        sample_size= sample_outcome.shape[0]

        # Posterior Mean
        unobs_confounder_stats[tcase,0]= ( alpha + tcase + sample_size*np.mean(sample_outcome))/(sample_size+1)
        # Posterior Vairance
        unobs_confounder_stats[tcase,1]= eps/(sample_size+1)
        print('Alpha: ', alpha, ' Sample_size : ', sample_size, ' Sum_ ', np.mean(sample_outcome), np.sum(sample_outcome), unobs_confounder_stats[tcase,0])
#     print('Posterior Distirbution: ', unobs_confounder_stats)

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
#     print( 'U Shape', unobs_confounder.shape )
#     print( 'U Val', unobs_confounder[ treatment == 0 ][:20],  unobs_confounder[ treatment == 1 ][:20] )
    corr_treat= pearsonr( unobs_confounder, treatment )
    corr_out= pearsonr( unobs_confounder, outcome )
#     print( 'Correlation Treatment: ', corr_treat )
#     print( 'Correlation_Outcome:' , corr_out )

    # Add the new column to the datatframe
    unobs_confounder= np.reshape( unobs_confounder, (unobs_confounder.shape[0], 1) )
    X_data_unobs_conf= np.concatenate((X_data, unobs_confounder), axis=1)
    
    return corr_treat[0], corr_out[0], X_data_unobs_conf


def refutation_analysis(method, case):
    '''
        method: lr; svm
    '''
    # Treatment: A_data; Features: X_data; Labels: Y_data [ Each of shape 10k*1 or 10k*50 ]
    # A and A_data are the same

    xDim = 50
    nSim = 10000

    #Data Generation
    A_matched_Z = []
    RMSEs = []

    # A percentage Z match 52% params
    # p_AgivenZ= 0.6
    # p_AgivenNotZ=0.5

    # A percentage Z match 77% params
    p_AgivenZ= 0.8
    p_AgivenNotZ=0.2

    X_data, Y_data, A_data, nObs, Group_data, Y_0_data, Y_1_data, Z, A = sim_Data(xDim, nSim, p_AgivenZ, p_AgivenNotZ)
    print(np.sum(Y_0_data), np.sum(Y_1_data))

    #Results on Normal Data
    if method == 'IPTW_LR':
        Yhat_0, Yhat_1 = fit_IPTW_LR(X_data, Y_data, A_data, nObs)
    elif method == 'IPTW_SVM':
        Yhat_0, Yhat_1 = fit_IPTW_SVM(X_data, Y_data, A_data, nObs)
        
    a_matched_z, rmse = evaluation(Yhat_0, Yhat_1, Group_data, Y_0_data, Y_1_data, A, Z)
    print('% of A matched Z', a_matched_z)

    A_matched_Z.append(a_matched_z)
    RMSEs.append(rmse)

    #UnObs Confounder
    A_matched_Z_unobs = []
    RMSEs_unobs = []

    corr_t=[]
    corr_y=[]
    alpha_range= [10**3, 10**4, 10**5]

    for alpha in alpha_range:

        #Generate Obs Refutation Data
        eps= 5000*alpha
        if alpha ==10**3:
            eps=5*eps

        corr_treat, corr_out, X_data_unobs= bayes_unobs_confounder(X_data, A_data, Y_data, alpha, eps)
        corr_t.append(corr_treat)
        corr_y.append(corr_out)

        #Results on Confounded Data
        Yhat_0, Yhat_1 = fit_IPTW_LR(X_data_unobs,Y_data,  A_data, nObs)
        a_matched_z, rmse = evaluation(Yhat_0, Yhat_1, Group_data, Y_0_data, Y_1_data, A, Z)

        A_matched_Z_unobs.append(a_matched_z)
        RMSEs_unobs.append(rmse)

        print('Final')
        print('Correlation Treatment: ', corr_treat)
        print('Correlation Outcome: ', corr_out)

    alpha_range=np.array(alpha_range)
    alpha_range=10000*(1/alpha_range)

    A_matched_Z_unobs = np.array(A_matched_Z_unobs)
    RMSEs_unobs = np.array(RMSEs_unobs)

    sort_indice = np.argsort(A_matched_Z_unobs)
    sort_indice = sort_indice[::-1]
    print(sort_indice, RMSEs_unobs)
    
    return alpha_range, sort_indice, a_matched_z, A_matched_Z_unobs, RMSEs_unobs, RMSEs, corr_t, corr_y

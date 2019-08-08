# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:43:50 2019

@author: sj2118
"""

#import pdb
import os, sys, time, pickle, copy, h5py
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import keras
from keras import backend as K
import tensorflow as tf

# =============================================================================
# # GPU code for session activation
# config = tf.ConfigProto()
# #config.gpu_options.per_process_gpu_memory_fraction = 0.4 #what portion of gpu to use
# session = tf.Session(config=config)
# K.set_session(session)
# =============================================================================
#################################################

#try: K.tensorflow_backend._get_available_gpus()
#except: print("Failed to create session..trying again")
#K.tensorflow_backend._get_available_gpus()

from keras.callbacks import EarlyStopping, Callback
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.constraints import max_norm
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import scale
import optunity
import lifelines.utils
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter


"""----Loading in files [Training and Test]----"""
#os.chdir(os.path.join(os.path.expanduser('~'),'P:/myWS'))

#import input data: i_orig=list of patient IDs, y_orig=censoring status and survival times for patients, x_orig=input data for patients (i.e. motion descriptors [11,514-element vector])
with open('3dall_data.pkl', 'rb') as f: x3Ddat,ydat,idat,covdat,cmrvolsdat  = pickle.load(f)
numcovs  = covdat.shape[1]+cmrvolsdat.shape[1]
xdat = np.concatenate((x3Ddat,covdat,cmrvolsdat), axis=-1)
inpshape = x3Ddat.shape[1]

nbPatient = 302


def prepare_data(x, e, t, c):
    return (x.astype("float32"), e.astype("int32"), t.astype("float32"), c.astype("float32"))


def sort4minibatches(xvals, evals, tvals, cvals, batchsize):
    ntot = len(xvals)
    indices = np.arange(ntot)
    np.random.shuffle(indices)
    start_idx=0
    esall = []
    for end_idx in list(range(batchsize, batchsize*(ntot//batchsize)+1, batchsize))+[ntot]:
        excerpt = indices[start_idx:end_idx]
        sort_idx = np.argsort(tvals[excerpt])[::-1]
        es = excerpt[sort_idx]
        esall += list(es)
        start_idx = end_idx
    return (xvals[esall], evals[esall], tvals[esall], cvals[esall], esall)

def normz(x):
    return scale(x, axis=0, with_mean=True, with_std=True)


#Define Cox PH partial likelihood function loss.
#Arguments: E (censoring status), risk (risk [log hazard ratio] predicted by network) for batch of input subjects
#As defined, this function requires that all subjects in input batch must be sorted in descending order of survival/censoring time (i.e. arguments E and risk will be in this order)
def _negative_log_likelihood(Ind, Prob):
    return -K.sum(K.log(K.sum(Ind * Prob, axis = 1)))


def get_Prob(e, tm, nbInt):
    e = e.reshape((nbPatient, ))
    tm = tm.reshape((nbPatient, ))
    t_max = np.max(tm)
    gap = t_max / (nbInt - 1)
    interval = np.ceil(tm / gap)
    
    indicator = np.zeros((nbPatient, nbInt))
    for i in np.array(range(0, nbPatient)):
        if(e[i] == 1):
            tmp = np.zeros((nbInt, ))
            tmp[(interval[i].astype("int32") - 1)] = 1
            indicator[i] = tmp
        else:
            tmp = np.ones((nbInt, ))
            tmp[:(interval[i].astype("int32"))] = np.zeros(((interval[i].astype("int32")), ))
            indicator[i] = tmp
    
    return indicator


def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G


def weighted_c_index(Y_train, T_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        tmp_idx = np.where(G[0,:] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1./G[1, -1])**2
        else:
            W = (1./G[1, tmp_idx[0]])**2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1. # give weights

        if (T_test[i]<=Time and Y_test[i]==1):
            N_t[i,:] = 1.

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result



#------define architecture and fitting
def modfit(xtr, ytr, alpha, dro, units1, units2, units3, l1r, lr, batchsize, numepochs):
    X_tr, E_tr, TM_tr, C_tr = prepare_data(xtr[:,:-numcovs], ytr[:,0,np.newaxis], ytr[:,1], xtr[:,-numcovs:])

    #Arrange data into minibatches (based on specified batch size), and within each minibatch, sort in descending order of survival/censoring time (see explanation of Cox PH loss function definition)
    X_tr, E_tr, TM_tr, C_tr, _ = sort4minibatches(normz(X_tr), E_tr, TM_tr, normz(C_tr), batchsize)
    
    #before defining network architecture, clear current computation graph (if one exists), and specify input dimensionality
    K.clear_session()
#    inpshape = xtr.shape[1]
    
    #Specify Model Architecture
    inputvec= Input(shape=(inpshape,), name = 'inputvec')
    inputconv = Input(shape=(numcovs,), name = 'inputconv')
    x       = Dropout(dro, input_shape=(inpshape,))(inputvec)
    x       = Dense(units=int(units1), activation='relu', activity_regularizer=l1(10**l1r))(x)
    encoded = Dense(units=int(units2), activation='relu', name='encoded')(x)
    
    merge_layer = keras.layers.concatenate([encoded, inputconv])
    riskpred= Dense(units=units3,  activation='softmax', name='predicted_risk')(merge_layer)
    z       = Dense(units=int(units1),  activation='relu')(encoded)
    decoded = Dense(units=inpshape, activation='linear', name='decoded')(z)
    #Define model inputs (11,514-element motion descriptor vector) and outputs (reconstructed input and predicted risk) 
    model = Model(inputs=[inputvec, inputconv], outputs=[decoded,riskpred])
    model.summary()
    
    #Model compilation
    optimdef = Adam(lr = lr)
    
    model.compile(loss=[keras.losses.mean_squared_error, _negative_log_likelihood], loss_weights=[alpha,1-alpha], optimizer=optimdef, metrics={'decoded':keras.metrics.mean_squared_error})
    
    ##Indicator for the probability
    Ind_prob = get_Prob(E_tr, TM_tr, units3)
    
    #Run model
    mlog = model.fit([X_tr, C_tr], [X_tr, Ind_prob], batch_size=batchsize, epochs=numepochs, shuffle=False, verbose=1)
    del mlog
    
# =============================================================================
#     pdb.set_trace()
#     weight = model.layers[0].get_weights()
#     print(weight)
#     weight = model.layers[1].get_weights()
#     print(weight)
#     weight = model.layers[2].get_weights()
#     print(weight)
#     weight = model.layers[3].get_weights()
#     print(weight)
#     weight = model.layers[4].get_weights()
#     print(weight)
#     weight = model.layers[5].get_weights()
#     print(weight)
#     weight = model.layers[6].get_weights()
#     print(weight)
#     weight = model.layers[7].get_weights()
#     print(weight)
# =============================================================================

    return model


def chimpred(m_MLP, x, y, reps=True):
    Xdat = normz(x[:,:-numcovs])
    Cdat = normz(x[:,-numcovs:])
    if reps==True:
        coll = []
        for _ in range(2): coll.append(m_MLP.predict([Xdat, Cdat], batch_size=1))
        preds = np.concatenate(coll, axis=-1).mean(axis=1)
    else: 
        preds = m_MLP.predict([Xdat, Cdat], batch_size=1)[1]
    C = concordance_index(y[:,1], -preds, y[:,0])
    return preds, C


def hypersearch_DL(x_data, y_data, method, nfolds, nevals, batch_size, num_epochs, alpha_range, dro_range, units1_range, units2_range, l1r_range, lrexp_range):
    @optunity.cross_validated(x=x_data, y=y_data, num_folds=nfolds)
    def modelrun(x_train, y_train, x_test, y_test, alpha, dro, units1, units2, l1r, lr):
        mod_MLP = modfit(xtr=x_train, ytr=y_train, alpha = alpha, dro = dro, units1 = units1, units2 = units2, l1r = l1r, lr = 10**lr, batchsize = batch_size, numepochs = num_epochs)
        _, cv_C = chimpred(mod_MLP, x_test, y_test,reps=False)
        return cv_C
    optimal_pars, searchlog, _ = optunity.maximize(modelrun, num_evals=nevals, solver_name=method, alpha=alpha_range, lr=lrexp_range, l1r=l1r_range, units1 = units1_range, units2= units2_range, dro=dro_range)
    print('Optimal hyperparameters : ' + str(optimal_pars))
    print('Cross-validated C after tuning: %1.3f' % searchlog.optimum)
    return optimal_pars, searchlog

preds_full = []
preds_bootfull = []
preds_boot = []
inds_inbag = []
Cb_opts  = []


preds_full_cox = []
preds_bootfull_cox = []
preds_boot_cox = []
Cb_opts_cox = []

#STEP 1
#(1a) find optimal hyperparameters
#opars, osummary = hypersearch_DL(x_data=xdat, y_data=ydat, method='particle swarm', nfolds=6, nevals=50, batch_size=16, num_epochs=100, lrexp_range=[-6.,-4.5], l1r_range=[-7,-4], dro_range=[.1,.9], units1_range=[75,250], units2_range=[5,20], alpha_range=[0.3,0.7])
with open('output.pkl', 'rb') as f: inputdata_list=pickle.load(f)
opars = inputdata_list[10]
del inputdata_list

#(1b) using optimal hyperparameters, train a model on full sample
omMLP = modfit(xtr = xdat, ytr = ydat, alpha = opars['alpha'], dro = opars['dro'], units1=opars['units1'], units2=opars['units2'], units3 = 40, lr=10**opars['lrexp'], l1r=opars['l1r'], batchsize=16, numepochs=100)

# =============================================================================
# #(1c) Compute Harrell's Concordance index
# predfull, C_app = chimpred(omMLP, xdat, ydat, False)
# preds_full.append(predfull)
# print('Apparent concordance index = {0:.4f}'.format(C_app))
# 
# #print('now fitting Cox....')
# 
# #omCox = coxreg_single_run(xdat, ydat, penalty=opars['penalty'])
# #print('now predicting for Cox...')
# #predfull_cox, C_app_cox = coxpred(omCox, xdat, ydat)
# #preds_full_cox.append(predfull_cox)
# 
# print('done with app step')
# 
# 
# #BOOTSTRAP SAMPLING
# 
# #define useful variables
# nsmp = len(xdat)
# rowids = [_ for _ in range(nsmp)]
# B = 100
# for b in range(B):
#     print('\n-------------------------------------')
#     print('Current bootstrap sample:', b, 'of', B-1)
#     print('---------------------------------------')
# 
#     #STEP 2: Generate a bootstrap sample by doing n random selections with replacement (where n is the sample size)
#     b_inds = np.random.choice(rowids, size=nsmp, replace=True)
# #    b_inds = dlinbag[b]
#     xboot = xdat[b_inds]
#     yboot = ydat[b_inds]
# 
#     #(2a) find optimal hyperparameters
#     bpars, bsummary = hypersearch_DL(x_data=xdat, y_data=ydat, method='particle swarm', nfolds=6, nevals=50, batch_size=16, num_epochs=100, lrexp_range=[-6.,-4.5], l1r_range=[-7,-4], dro_range=[.1,.9], units1_range=[75,250], units2_range=[5,20], alpha_range=[0.3,0.7])
#     
#     #(2b) using optimal hyperparameters, train a model on bootstrap sample
#     bmMLP = modfit(xtr=xboot, ytr=yboot, alpha = bpars['alpha'], dro = bpars['dro'], units1=bpars['units1'], units2=bpars['units2'], lr=10**bpars['lr'], l1r=bpars['l1r'], batchsize=16, numepochs=100)
#     #bmCox = coxreg_single_run(xboot, yboot, penalty=bpars['penalty'])    
#     #(2c[i])  Using bootstrap-trained model, compute predictions on bootstrap sample. Evaluate accuracy of predictions (Harrell's Concordance index)
#     predboot, Cb_boot     = chimpred(bmMLP, xboot, yboot, False)
#     #predboot_cox, Cb_boot_cox = coxpred(bmCox, xboot, yboot)
#     #(2c[ii]) Using bootstrap-trained model, compute predictions on FULL sample.     Evaluate accuracy of predictions (Harrell's Concordance index)
#     predbootfull, Cb_full = chimpred(bmMLP, xdat, ydat, False)
#     #predbootfull_cox, Cb_full_cox = coxpred(bmCox, xdat, ydat)
#     #STEP 3: Compute optimism for bth bootstrap sample, as difference between results from 2c[i] and 2c[ii]
#     Cb_opt = Cb_boot - Cb_full
#     #Cb_opt_cox = Cb_boot_cox - Cb_full_cox    
#     #store data on current bootstrap sample (predictions, C-indices)
#     preds_boot.append(predboot)
#     #preds_boot_cox.append(predboot_cox)
#     preds_bootfull.append(predbootfull)
#     #preds_bootfull_cox.append(predbootfull_cox)
#     inds_inbag.append(b_inds)
#     Cb_opts.append(Cb_opt)
#     #Cb_opts_cox.append(Cb_opt_cox)
#     del bmMLP
# 
# 
# #STEP 5
# #Compute bootstrap-estimated optimism (mean of optimism estimates across the B bootstrap samples)
# C_opt = np.mean(Cb_opts)
# 
# #Adjust apparent C using bootstrap-estimated optimism
# C_adj = C_app - C_opt
# 
# #compute confidence intervals for optimism-adjusted C
# C_opt_95confint = np.percentile([C_app - o for o in Cb_opts], q=[2.5, 97.5])
# 
# #save summary of results
# print('\n\n======SUMMARY - TRAINING & VALIDATION FOR DEEP LEARNING MODEL======\n')
# print('Apparent concordance index = {0:.4f}\n'.format(C_app))
# print('Optimism bootstrap estimate = {0:.4f}\n'.format(C_opt))
# print('Optimism-adjusted concordance index = {0:.4f}, and 95% CI = {1}\n'.format(C_adj, C_opt_95confint))
# 
# with open('my3dall_new_output111.pkl', 'wb') as f: pickle.dump([idat, ydat, preds_full, preds_boot, preds_bootfull, inds_inbag, C_app, Cb_opts, opars],f)
# 
# 
# 
# =============================================================================

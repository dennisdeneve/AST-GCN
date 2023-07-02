#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:39:22 2018

@author: lhfcitylab
"""

import matplotlib.pyplot as plt

'''
This method is used to plot the test results and the true labels for all the test data as well as for a single day. 
'''
def plot_result(test_result,test_label1,path):
    ##all test result visualization
    # Creates a figure with size (7, 1.5) for plotting the all test result.
    fig1 = plt.figure(figsize=(7,1.5))
    #    ax1 = fig1.add_subplot(1,1,1)
    #Extracts the predicted values and true labels from the test results and test labels.
    a_pred = test_result[:,0]
    a_true = test_label1[:,0]
    # Plots the predicted values in red and true labels in blue.
    plt.plot(a_pred,'r-',label='prediction')
    plt.plot(a_true,'b-',label='true')
    # Adds a legend to the plot with a fontsize of 10.
    plt.legend(loc='best',fontsize=10)
    # Saves the figure as a .jpg file in the path specified.
    plt.savefig(path+'/test_all.jpg')
    #    plt.show()

    ## oneday test result visualization
    # Creates another figure with size (7, 1.5) for plotting the one day test result.
    fig1 = plt.figure(figsize=(7,1.5))
    #    ax1 = fig1.add_subplot(1,1,1)
    # Extracts the predicted values and true labels for the first 96 time steps (one day).
    a_pred = test_result[0:96,0]
    a_true = test_label1[0:96,0]
    # Plots the predicted values in red and true labels in blue.
    plt.plot(a_pred,'r-',label="prediction")
    plt.plot(a_true,'b-',label="true")
    # Adds a legend to the plot with a fontsize of 10.
    plt.legend(loc='best',fontsize=10)
    # Saves the figure as a .jpg file in the path specified.
    plt.savefig(path+'/test_oneday.jpg')
    #    plt.show()
    
    
#def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,test_mape,test_smape,path):
    ###train_rmse & test_rmse 
    # Plot train_rmse and test_rase
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.jpg')
    #plt.show()

    #### train_loss & train_rmse
    # Plot train loss
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.jpg')
    #plt.show()
    #plt.close
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_loss[150:],'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss[150:].jpg')
    #plt.show()
    
    ## train_rmse
    ##Plot train rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.jpg')
    #plt.show()
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(train_rmse[150:],'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse[150:].jpg')
    #plt.show()

    ##### test_accuracy
    ## Plot accuracy graph
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.jpg')
    #plt.show()
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_acc[150:], 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc[150:].jpg')
    #plt.show()

    ### rmse
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.jpg')
   # plt.show()
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_rmse[150:], 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse[150:].jpg')
    #plt.show()

    ### mae
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.jpg')
    #plt.show()
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mae[150:], 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae[150:].jpg')
    #plt.show()

    # MAPE
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mape, 'b-', label="test_mape")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mape.jpg')
    #plt.show()
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_mape[150:], 'b-', label="test_mape")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mape[150:].jpg')
    #plt.show()

     # SMAPE
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_smape, 'b-', label="test_smape")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_smape.jpg')
#    plt.show()
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(test_smape[150:], 'b-', label="test_smape")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_smape[150:].jpg')
#    plt.show()


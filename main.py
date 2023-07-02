# -*- coding: utf-8 -*-
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import argparse
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from Model.acell import preprocess_data,load_assist_data
from Model.tgcn import TGCN
from Vis.visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
import time
import yaml

time_start = time.time()

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to YAML config file')
args = parser.parse_args()

# Load the YAML config file which contains all the required settings for platform
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_name = config['model_name']['default']
noise_name = config['noise_name']['default']
data_name = config['dataset']['default']
train_rate = config['train_rate']['default']
seq_len =  config['seq_len']['default']
pre_len =  config['pre_len']['default']
batch_size =  config['batch_size']['default']
lr =  config['learning_rate']['default']
training_epoch =  config['training_epoch']['default']
gru_units =  config['gru_units']['default']
dim =  config['dim']['default']
scheme =  config['scheme']['default']
PG =  config['noise_param']['default']

########## load data #########
if data_name == 'sz':
    data, adj = load_assist_data('sz')
#if data_name == 'sh':
#    data, adj = load_sh_data('sh')

### Perturbation Analysis
def MaxMinNormalization(x,Max,Min):
    x = (x-Min)/(Max-Min)
    return x

## Applying different types of noise filter to data
if noise_name == 'Gauss':
    #If noise_name is 'Gauss', 
    # it generates Gaussian noise using the np.random.normal() function with mean 0 and standard deviation PG. 
    # The generated noise is normalized using MaxMinNormalization() and added to the input data.
    Gauss = np.random.normal(0,PG,size=data.shape)
    noise_Gauss = MaxMinNormalization(Gauss,np.max(Gauss),np.min(Gauss))
    data = data + noise_Gauss
elif noise_name == 'Possion':
    #If noise_name is 'Possion',
    # it generates Poisson noise using the np.random.poisson() function with mean PG. 
    # The generated noise is also normalized and added to the input data.
    Possion = np.random.poisson(PG,size=data.shape)
    noise_Possion = MaxMinNormalization(Possion,np.max(Possion),np.min(Possion))
    data = data + noise_Possion
else:
    # Else data is unchanged
    data = data

# After adding noise to the data, time_len and num_nodes are calculated based on the shape of the data. 
# Finally, data1 is created as a NumPy matrix with dtype=np.float32.
time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)

#### normalization
max_value = np.max(data1)
data1  = data1/max_value

data1.columns = data.columns

# Distinguishing the various model types
if model_name == 'ast-gcn':
    if scheme == 1:
        name = 'add poi dim'
    elif scheme == 2:
        name = 'add weather dim'
    else:
        name = 'add poi + weather dim'
else:
    name = 'tgcn'


print("Starting the evaluation process  :)")
print('model:', model_name)
print('scheme:', name)
print('noise_name:', noise_name)
print('noise_param:', PG)

trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len, model_name, scheme)

# print(trainX)
# print(trainY)
# print(testX)
# print(testY)

totalbatch = int(trainX.shape[0]/batch_size)
print("The size of dataset is: ", str(batch_size))
training_data_count = len(trainX)




    
    
    
if model_name == 'ast-gcn':
    if scheme == 1:
        inputs = tf.keras.Input(shape=[seq_len+1, num_nodes], dtype=tf.float32)
    elif scheme == 2:
        inputs = tf.keras.Input(shape=[seq_len*2+pre_len, num_nodes], dtype=tf.float32)
    else:
        inputs = tf.keras.Input(shape=[seq_len*2+pre_len+1, num_nodes], dtype=tf.float32)

else:
    inputs = tf.keras.Input(shape=[seq_len, num_nodes], dtype=tf.float32)

labels = tf.keras.Input(shape=[pre_len, num_nodes], dtype=tf.float32)

############ Graph weights defined ############
# The weights are defined as a dictionary named 'weights', 
# where the key 'out' maps to a TensorFlow Variable representing the weight matrix 
# that will be applied to the output of the TGCN model. 
# The weight matrix has a shape of [gru_units, pre_len], where 'gru_units' is the number of GRU units in the TGCN cell 
# and 'pre_len' is the number of time steps to predict into the future. 
# The values in the weight matrix are randomly initialized from a normal distribution with mean=1.0.
weights = {
    'out': tf.Variable(tf.random.normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random.normal([pre_len]),name='bias_o')}

#The TGCN model is then called with the inputs, weights, and biases as arguments, 
# and the output of the model is stored in the variable 'pred'. 
# Finally, the predicted values are stored in 'y_pred', which will be used for training and evaluation of the model.
pred,ttts,ttto = TGCN(inputs, weights, biases)
y_pred = pred
      

########### optimizer used to train the model ############
# lambda_loss is a hyperparameter that controls the strength of the L2 regularization applied to the trainable variables in the model.
lambda_loss = 0.0015
#Lreg is the L2 regularization term, which is computed as the sum of the L2 norms of all the trainable variables 
#in the model multiplied by the lambda_loss hyperparameter.
Lreg = lambda_loss * sum(tf.compat.v1.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
# labels is the ground truth data.
label = tf.compat.v1.reshape(labels, [-1,num_nodes])
# loss is the mean squared error loss between y_pred and labels, with an added regularization term.
print('y_pred_shape:', y_pred.shape)
print('label_shape:', label.shape)
loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.l2_loss(y_pred-label) + Lreg)
##rmse -> is the root mean squared error between y_pred and labels.
error = tf.compat.v1.sqrt(tf.compat.v1.reduce_mean(tf.compat.v1.square(y_pred-label)))
# optimizer is the Adam optimizer, which is used to minimize the loss function during training. 
# The learning rate used by the optimizer is specified by the lr variable.
optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)


###### Initialize session ######
## initializes a TensorFlow session with GPU options set, and then initializes 
# the global variables in the graph. It also creates a Saver object for saving and restoring the model variables.
variables = tf.compat.v1.global_variables()
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())  
#sess = tf.Session()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
sess.run(tf.compat.v1.global_variables_initializer())

# It then creates a path to save the model using various parameters and creates the directory if it doesn't exist.
#out = 'out/%s'%(model_name)
out = 'out/%s_%s'%(model_name,noise_name)
path1 = '%s_%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r_scheme%r_PG%r'%(model_name,name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch,scheme,PG)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
###### evaluation ###### 
def evaluation(a, b):
    """
    Calculate evaluation metrics for comparing actual values 'a' and predicted values 'b'.
    a: actual values
    b: predicted values
    """
    # RMSE and MAE calculations
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    
    # MAPE calculation
    mape = np.zeros_like(a)
    idx = a != 0
    mape[idx] = np.abs((a[idx] - b[idx]) / a[idx]) * 100
    
    # SMAPE calculation
    smape = np.zeros_like(a)
    idx = (a != 0) & (b != 0)
    smape[idx] = np.abs(a[idx] - b[idx]) / ((np.abs(a[idx]) + np.abs(b[idx])) / 2) * 100
    
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    
    return rmse, mae, np.mean(mape), np.mean(smape), 1 - F_norm, r2, var


# Initialising all the variables
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_mape,test_smape,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[],[],[]
  
## This is the training loop of the model. 
# The loop iterates over the specified number of epochs and in each epoch, 
# the training set is split into mini-batches and fed to the model for training.
for epoch in range(training_epoch):
    # The optimizer is run on each mini-batch, and the loss and error metrics are calculated. 
    # The test set is evaluated completely at every epoch, and the loss and error metrics are calculated. 
    
    
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})

    # The evaluation metrics such as RMSE, MAE, MAPE, and SMAPE are calculated on the test set, and the results are stored. 
    testoutput = np.abs(test_output)
    test_label = np.reshape(testY,[-1,num_nodes])
    rmse, mae,mape,smape, acc, r2_score, var_score = evaluation(test_label, testoutput)
    test_label1 = test_label * max_value
    test_output1 = testoutput * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_mape.append(mape * max_value)
    test_mape.append(smape * max_value)
    test_acc.append(acc)
    
    
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    
    # The training and testing progress is printed after every epoch. 
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          
          'test_loss:{:.4}'.format(loss2),
          
          'test_rmse:{:.4}'.format(rmse),
          'test_mae:{:.4}'.format(mae),
          'test_mape:{:.4}'.format(mape),
          'test_smape:{:.4}'.format(smape),
          'test_acc:{:.4}'.format(acc))
    
    # The model is also saved every 500 epochs - reduced to 10 for now
    if (epoch % 10 == 0):        
        saver.save(sess, path+'/model_100/ASTGCN_pre_%r'%epoch, global_step = epoch)


# End timer and print it out
time_end = time.time()
print("Timer ended : " ,time_end-time_start,'s')


##############  visualizing the results of the model after training.###############
# First code computes the training and batch RMSE and loss values by averaging the RMSE and 
# loss for each batch in the training data. 
#x = [i for i in range(training_epoch)]
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
#test_rmse = [float(i) for i in test_rmse]

#Then, it finds the index of the test RMSE value with the minimum value and saves the corresponding 
# predicted results to a CSV file.
index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/test_result.csv',index = False,header = False)

# It then uses the plot_result and plot_error functions to plot the predicted and actual values, 
# as well as the training and test RMSE, loss, accuracy, MAE, and MAPE
plot_result(test_result,test_label1,path)
plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,test_mape,test_smape,path)


evalution = []
evalution.append(np.min(test_rmse))
evalution.append(test_mae[index])
evalution.append(test_mape[index])

print("length of smape ", smape)
print("length of index ", index)

# This will append np.nan (a special value indicating "Not a Number") to the 
# evalution list if the index is out of range, instead of raising an IndexError.
if index < len(test_smape):
    evalution.append(test_smape[index])
else:
    evalution.append(np.nan)

#evalution.append(test_smape[index])

evalution.append(test_acc[index])
evalution.append(test_r2[index])
evalution.append(test_var[index])
evalution = pd.DataFrame(evalution)

# Saving the evaluation to a .csv file, and prints out all the details again
evalution.to_csv(path+'/evalution.csv',index=False,header=None)
print('successfully saved evaluation to csv file : ',path)
print('model_name:', model_name)
print('scheme:', scheme)
print('name:', name)
print('noise_name:', noise_name)
print('PG:', PG)
print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      #'min_smape:%r'%(test_smape[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index])
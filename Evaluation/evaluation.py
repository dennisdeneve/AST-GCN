# eval
import numpy as np
import pandas as pd
from Vis.visualization import plot_result,plot_error

def eval(batch_rmse, totalbatch, batch_loss , test_rmse, test_pred, path, 
        test_acc, test_mae, test_mape, test_smape,
        test_r2, test_var, test_label1
        ):
    b = int(len(batch_rmse)/totalbatch)
    batch_rmse1 = [i for i in batch_rmse]
    train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
    batch_loss1 = [i for i in batch_loss]
    train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
    #test_rmse = [float(i) for i in test_rmse]

    #Then, it finds the index of the test RMSE value with the minimum value and saves the corresponding  predicted results to a CSV file.
    index = test_rmse.index(np.min(test_rmse))

    test_result = test_pred[index]
    var = pd.DataFrame(test_result)
    var.to_csv(path+'/test_result.csv',index = False,header = False)
    

     ##############  visualizing the results of the evaluation of model .###############
    # It then uses the plot_result and plot_error functions to plot the predicted and actual values, 
    # as well as the training and test RMSE, loss, accuracy, MAE, and MAPE
    plot_result(test_result,test_label1,path)
    plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,test_mape,test_smape,path)

    evalution = []
    evalution.append(np.min(test_rmse))
    evalution.append(test_mae[index])
    evalution.append(test_mape[index])

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
    
    print('min_rmse:%r'%(np.min(test_rmse)),
        'min_mae:%r'%(test_mae[index]),
        #'min_smape:%r'%(test_smape[index]),
        'max_acc:%r'%(test_acc[index]),
        'r2:%r'%(test_r2[index]),
        'var:%r'%test_var[index])
                    
                

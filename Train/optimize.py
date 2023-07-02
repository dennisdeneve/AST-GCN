#optimize 
import tensorflow as tf

def optimize(labels, num_nodes,y_pred,lr):
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
    return optimizer

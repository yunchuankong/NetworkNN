###
## Only conduct one time trainging/testing for a dataset
###

from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics
from random import seed
import time

## timer
# start_time = time.time()
# print("Total time used: %s minutes " % ((time.time() - start_time)/60) )

tf.reset_default_graph()

expression = np.loadtxt("C:/Users/yunchuan/Dropbox/Research_Yu/kingdom/data_expression_sim.csv", dtype=float, delimiter=",", skiprows=1)
expression = np.array(expression[:,:-1])
label_vec = np.array(expression[:,-1], dtype=int)
labels = []
for l in label_vec:
    if l == 1:
        labels.append([0,1])
    else:
        labels.append([1,0])
labels = np.array(labels,dtype=int)

expression, labels = shuffle(expression, labels) ## different here from rfnn.py
x_train = expression[:320, :]
x_test = expression[320:, :]
y_train = labels[:320, :]
y_test = labels[320:, :]

## hyper-parameters and settings
L2 = False
droph1 = False
learning_rate = 0.0001
# learning_rate2 = 0.0001
training_epochs = 110
# trials = 1
# test_prop = 0.2
batch_size = 8
display_step = 1

n_hidden_1 = 1024
n_hidden_2 = 64
n_hidden_3 = 16
# n_hidden_4 = 16
n_classes = 2
n_features = np.shape(expression)[1]

## initiate training logs
loss_rec = np.zeros([training_epochs, 1])
training_eval = np.zeros([training_epochs, 2])
# testing_eval = np.zeros([int(training_epochs/10), 2])
# avg_test_acc = 0.
# avg_test_auc = 0.

def multilayer_perceptron(x, weights, biases, keep_prob):

    # layer_1 = tf.add(tf.matmul(x, tf.multiply(weights['h1'], partition)), biases['b1'])
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.relu(layer_1)
    if droph1:
        layer_1 = tf.nn.dropout(layer_1, keep_prob=keep_prob)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    ## Do not use batch-norm
    # layer_3 = tf.contrib.layers.batch_norm(layer_3, center=True, scale=True,
    #                                   is_training=is_training)
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob=keep_prob)

    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)
    # layer_4 = tf.nn.dropout(layer_4, keep_prob=keep_prob)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
# is_training = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)

weights = {
    'h1': tf.Variable(tf.truncated_normal(shape=[n_features, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_hidden_3], stddev=0.1)),
    # 'h4': tf.Variable(tf.truncated_normal(shape=[n_hidden_3, n_hidden_4], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_3, n_classes], stddev=0.1))

}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    # 'b4': tf.Variable(tf.zeros([n_hidden_4])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
if L2:
    reg = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
          tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out'])
    cost = tf.reduce_mean(cost + 0.1 * reg)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

## Evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
y_score = tf.nn.softmax(logits=pred)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_batch = int(np.shape(x_train)[0] / batch_size)

    ## for monitoring weights
    # w1_pre = sess.run(weights['h1'][:10, :10], feed_dict={x: expression, y: labels, keep_prob: 1})

    ## Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        x_tmp, y_tmp = shuffle(x_train, y_train)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = x_tmp[i*batch_size:i*batch_size+batch_size], \
                                y_tmp[i*batch_size:i*batch_size+batch_size]

            # if epoch <= 69:
            _, c= sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
                                                        keep_prob: 0.9,
                                                        lr: learning_rate
                                                        })
            # else:
            #     _, c= sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
            #                                                 keep_prob: 0.9,
            #                                                 lr: learning_rate2
            #                                                 })
            # Compute average loss
            avg_cost += c / total_batch

        del x_tmp
        del y_tmp

        ## Display logs per epoch step
        # if epoch % display_step == 0:
            # loss_rec[epoch] = avg_cost
            # # print ("Epoch:", '%d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_train, y: y_train, keep_prob: 1})
            # auc = metrics.roc_auc_score(y_train, y_s)
            # training_eval[epoch] = [acc, auc]
            # print ("Epoch:", '%d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost),
                    # "Training accuracy:", round(acc,3), " Training auc:", round(auc,3))

    ## Testing cycle
    acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_test, y: y_test, keep_prob: 1})
    auc = metrics.roc_auc_score(y_test, y_s)
    # print("*****=====", "Testing accuracy: ", acc, " Testing auc: ", auc, "=====*****")
    print(auc)




    # avg_test_acc += acc / trials
    # avg_test_auc += auc / trials

    # print ("***** ============================== *****")
    # print ("Average Testing Accuracy: ", round(avg_test_acc,3))
    # print ("Average Testing AUC: ", round(avg_test_auc,3))
    # print ("***** ============================== *****")
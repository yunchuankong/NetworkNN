## v3 - plus variable importance

from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics
from random import seed
import time

start_time = time.time()
tf.reset_default_graph()

## load and prepare the data
# expression = np.loadtxt("../data_expression.csv", dtype=float, delimiter=",")
expression = np.loadtxt("../data_expression_sim.csv", dtype=float, delimiter=",")
label_vec = np.array(expression[:,-1], dtype=int)
expression = np.array(expression[:,:-1])
labels = []
for l in label_vec:
    if l == 1:
        labels.append([0,1])
    else:
        labels.append([1,0])
labels = np.array(labels,dtype=int)
# partition = np.loadtxt("../partition.txt", dtype=int, delimiter=None)
partition = np.loadtxt("../partition_sim.txt", dtype=int, delimiter=None)


## hyper-parameters and settings
# resume = False
L2 = True
learning_rate = 0.001
# learning_rate2 = 0.0001
training_epochs = 25
trials = 10
test_prop = 0.2
batch_size = 8
signal_to_noise = 0.04 ## anticipated s/n, for var importance display purpose
display_step = 1 ## never chanve this

n_hidden_1 = 1024
n_hidden_2 = 64
n_hidden_3 = 16
n_classes = 2

## prepare for training and testing
n_features = np.shape(expression)[1]
n_instances = np.shape(expression)[0]
print ("Number of instances: ", n_instances)
print ("Number of features: ", n_features)

def train_test_split(expression, labels):
    expression, labels = shuffle(expression, labels)
    expression_test = expression[:int(n_instances*test_prop),:]
    labels_test = labels[:int(n_instances*test_prop),:]
    expression_train = expression[int(n_instances*test_prop):,:]
    labels_train = labels[int(n_instances*test_prop):,:]
    return expression_train, labels_train, expression_test, labels_test

## initiate training logs
loss_rec = np.zeros([training_epochs, 1])
training_eval = np.zeros([training_epochs, 2])
# testing_eval = np.zeros([int(training_epochs/10), 2])
avg_test_acc = 0.
avg_test_auc = 0.

# def get_batch(x, y, batch_size):
#     batch_x, batch_y = shuffle(x, y, n_samples=batch_size)
#     return batch_x, batch_y


def multilayer_perceptron(x, weights, biases, keep_prob):

    # layer_1 = tf.add(tf.matmul(x, tf.multiply(weights['h1'], partition)), biases['b1'])
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
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

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

## Placeholders
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
# is_training = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)

weights = {
    'h1': tf.Variable(tf.truncated_normal(shape=[n_features, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_hidden_3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_3, n_classes], stddev=0.1))

}

# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'b3': tf.Variable(tf.random_normal([n_hidden_3])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
reg = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
      tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out'])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
if L2:
    cost = tf.reduce_mean(cost + 0.1 * reg)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

## Evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
y_score = tf.nn.softmax(logits=pred)

var_imp = np.zeros([trials+1, n_features])

# Launch the graph
with tf.Session() as sess:
    for t in range(int(trials)):
        seed(t)
        expression_train, labels_train, expression_test, labels_test = train_test_split(expression, labels)
        sess.run(tf.global_variables_initializer())
        total_batch = int(np.shape(expression_train)[0] / batch_size)

        ## for monitoring weights
        # w1_pre = sess.run(weights['h1'][:10, :10], feed_dict={x: expression, y: labels, keep_prob: 1})

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            expression_tmp, labels_tmp = shuffle(expression_train, labels_train)
            # Loop over all batches
            for i in range(total_batch-1):
                batch_x, batch_y = expression_tmp[i*batch_size:i*batch_size+batch_size], \
                                    labels_tmp[i*batch_size:i*batch_size+batch_size]

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

            del expression_tmp
            del labels_tmp

            # Display logs per epoch step
            if epoch % display_step == 0:
                loss_rec[epoch] = avg_cost
                # print ("Epoch:", '%d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                acc, y_s = sess.run([accuracy, y_score], feed_dict={x: expression_train, y: labels_train, keep_prob: 1})
                auc = metrics.roc_auc_score(labels_train, y_s)
                training_eval[epoch] = [acc, auc]
                print ("Epoch:", '%d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost),
                       "Training accuracy:", round(acc,3), " Training auc:", round(auc,3))
                # print ("Training accuracy:", acc, " Training auc:", auc)
                # w1 = sess.run(weights['h1'][:10, :10], feed_dict={x: expression, y: labels, keep_prob: 1})
                # print(w1 - w1_pre)
                # w1_pre = w1

        ## Testing cycle
        acc, y_s = sess.run([accuracy, y_score], feed_dict={x: expression_test, y: labels_test, keep_prob: 1})
        auc = metrics.roc_auc_score(labels_test, y_s)
        print("*****=====", t+1, "Testing accuracy: ", acc, " Testing auc: ", auc, "=====*****")
        avg_test_acc += acc / trials
        avg_test_auc += auc / trials

        ## variable importance evaluation cycle
        for g in range(n_features):
            expression_temp_g = expression_test
            d_g = abs(expression_temp_g[:,g]*0.05)
            expression_temp_g[:,g] += d_g
            y_s_g = sess.run(y_score, feed_dict={x: expression_temp_g, y: labels_test, keep_prob: 1})
            var_imp[t, g] = np.mean(abs(y_s_g[:, 1] - y_s[:, 1])/d_g)
            del expression_temp_g
            del d_g
            del y_s_g
        print("*****=====", t+1, "Variable importance recorded.", "=====*****")

    var_imp[-1, :] = np.mean(var_imp[:-1, :], axis=0)
    important_features = np.argsort(var_imp[-1, :])[:int(signal_to_noise/(signal_to_noise+1)*n_features)] + 1
    np.savetxt(str(n_instances) + "s_" + str(n_features) + "g_" + "varImp.csv",
               np.array(important_features, dtype=int), delimiter=",")
    print("*****=====", "Important feratures saved.", "=====*****")
    print ("***** ============================== *****")
    print ("Average Testing Accuracy: ", round(avg_test_acc,3))
    print ("Average Testing AUC: ", round(avg_test_auc,3))
    print ("***** ============================== *****")



print("Total time used: %s minutes " % ((time.time() - start_time)/60))
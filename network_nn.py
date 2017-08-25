from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics
from random import seed
tf.reset_default_graph()

# load and prepare the data
expression = np.loadtxt("../data_expression.csv" ,dtype=float, delimiter=",")
label_vec = np.array(expression[:,-1], dtype=int)
expression = np.array(expression[:,:-1])
n_features = np.shape(expression)[1]
n_instances = np.shape(expression)[0]
labels = []
for l in label_vec:
    if l == 1:
        labels.append([0,1])
    else:
        labels.append([1,0])
labels = np.array(labels,dtype=int)

partition = np.loadtxt("../partition.txt" ,dtype=int, delimiter=None)

expression, labels = shuffle(expression, labels)
expression_test = expression[:int(n_instances/5),:]
labels_test = labels[:int(n_instances/5),:]
expression = expression[int(n_instances/5):,:]
labels = labels[int(n_instances/5):,:]

resume = False
learning_rate = 0.001
training_epochs = 100
batch_size = 10
display_step = 1

n_hidden_1 = np.shape(partition)[1]
n_hidden_2 = 64
n_hidden_3 = 16
n_classes = 2

# def get_batch(x, y, batch_size):
#     batch_x, batch_y = shuffle(x, y, n_samples=batch_size)
#     return batch_x, batch_y

def make_1d_weights(indication):
    # w = [tf.Variable(tf.random_normal([1, 1])) if item == 1 else tf.zeros([1, 1]) for item in indication]
    w = [tf.Variable(tf.random_normal([1, 1])) if item == 1
         else tf.Variable(tf.zeros([1, 1]), trainable=False) for item in indication]
    return tf.reshape(w, [len(indication), 1])

def make_weights(n_hidden, partition):
    w = make_1d_weights(partition[:,0])
    for i in range(1, n_hidden):
        temp = make_1d_weights(partition[:,i])
        w = tf.concat([w, temp], 1)
    return w

def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.int32, [None, n_classes])

weights = {
    # 'h1': tf.Variable(tf.random_normal([n_features, n_hidden_1])),
    'h1': make_weights(n_hidden_1, partition),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Create a saver
saver = tf.train.Saver(max_to_keep=2)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
seed(1)
with tf.Session() as sess:
    sess.run(init)
    if resume:
        saver = tf.train.import_meta_graph('last_saved_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print ("Existing model loaded.")

    total_batch = int(np.shape(expression)[0] / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        expression_tmp, labels_tmp = shuffle(expression, labels)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = expression_tmp[i*batch_size:i*batch_size+batch_size], \
                               labels_tmp[i*batch_size:i*batch_size+batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c= sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            # Compute average loss
            avg_cost += c / total_batch

        del expression_tmp
        del labels_tmp

        # Display logs per epoch step
        if epoch % display_step == 0:
            # w = sess.run(weights['h1'][1:4,10:19], feed_dict={x: batch_x, y: batch_y})
            # print (w)
            print ("Epoch:", '%d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Training accuracy:", accuracy.eval({x: expression, y: labels}))

            y_score = tf.nn.softmax(logits=pred)
            auc = metrics.roc_auc_score(labels, y_score.eval({x: expression, y: labels}))
            print ("Training auc:", auc)


        if epoch % 10 == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("*** Testing accuracy:", accuracy.eval({x: expression_test, y: labels_test}))

            y_score = tf.nn.softmax(logits=pred)
            auc = metrics.roc_auc_score(labels_test, y_score.eval({x: expression_test, y: labels_test}))
            print("*** Testing auc:", auc)

            saver.save(sess, "last_saved_model")




# Name: Somayeh (Bahar) Shahsavarani
# Institute: University of Nebraska-Lincoln
# Department: Computer Science
# Advisor: Prof. Stephen Scott
# Updated: 11/12/2018
# This code was developed (as a part of my master thesis) to implement a vanilla convolutional neural network to classify speech emotions using spectrogram images.

import tensorflow as tf
import numpy as np

# read your training and test data
# training and test data are matrices with n rows of instances and m rows of features; the features are flattened
# train_x is the feature matrix for training
# train_y is the label matrix for training
# test_x is the feature matrix for test
# test_y is the label matrix for training

# number of classes depending on the number of target emotions in the database
n_classes

# batch size depending on the number of training data you have
batch_size = 512

# number of training epochs
hm_epochs = 4000

# height of the image
hm_pixels1 = 129

# width of the image
hm_pixels2 = 129

# define a placeholder for the feature matrices
x = tf.placeholder('float', [None, hm_pixels1 * hm_pixels2])

# define a placeholder for the label matrices
y = tf.placeholder('float')

# the dropout probability 
keep_prob = tf.placeholder(tf.float32)

# define convolutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# define pooling layer
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([10,10,1,8])),
                'W_conv2':tf.Variable(tf.random_normal([5,5,8,16])),
                'W_fc':tf.Variable(tf.random_normal([33*33*16,1024])),
                'out':tf.Variable(tf.random_normal([1024,n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([8])),
                'b_conv2':tf.Variable(tf.random_normal([16])),
                'b_fc':tf.Variable(tf.random_normal([1024])),
                'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1,hm_pixels1,hm_pixels2,1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 33*33*16])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])
    fc_drop = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc_drop,weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)  
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        for epoch in range(hm_epochs):
            epoch_loss = 0
                
            # build batches
            i = 0
            while i < len(train_x):
                start = i
                end   = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                    
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                epoch_loss += c
                i += batch_size
                
            training_accuracy = accuracy.eval({x:train_x, y:train_y, keep_prob: 1})
            test_accuracy = accuracy.eval({x:test_x, y:test_y, keep_prob: 1})
            loss_test = sess.run(loss,feed_dict = {x:test_x, y:test_y, keep_prob: 1})    


train_neural_network(x)



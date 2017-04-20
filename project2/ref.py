import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "/Users/wangrui/Desktop/SD/project#2/data/train.p"
testing_file = "/Users/wangrui/Desktop/SD/project#2/data/test.p"


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3])
print(y_train.shape[0])

n_train = len(train['features']) #Initially kept as 1000 for checking & correcting the syntax errors - Validation accuracy was too low

# TODO: Number of testing examples.
n_test = len(test['features'])

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

X_train=(255-X_train)/255 #Normalising the Training Data between 0 to 1
print(X_train[0].shape)

X_test=(255-X_test)/255 #Normalising the test data too.
print(X_test[0].shape)

from sklearn.utils import shuffle
import math as m


val_index=m.ceil(20*X_train.shape[0]/100); #Taking 20% Data for Validation
#test_index=m.ceil(10*X_train.shape[0]/100); #Taking 10% Data in the Training set as an undiluted Test set.
#This is needed only if there is not test data. Since we have the test.p as seperate test data, we neednot split the training data

end=X_train.shape[0]

X_train, y_train = shuffle(X_train, y_train)

X_validation,y_validation=X_train[0:val_index], y_train[0:val_index]
X_train,y_train=X_train[val_index+1:end], y_train[val_index+1:end]

print("Number of Validation samples =", X_validation.shape[0])
print("Number of Training samples =", X_train.shape[0])

import tensorflow as tf

EPOCHS = 5
BATCH_SIZE = 150
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = tf.constant(0,dtype=tf.float32)
    sigma = tf.constant(0.1,dtype=tf.float32)
    #
    weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 10], mean=mu, stddev=sigma)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 10, 25], mean=mu, stddev=sigma)),
    'wd1': tf.Variable(tf.truncated_normal([5*5*25, 120], mean=mu, stddev=sigma)),    
    'wd2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
    'wd3': tf.Variable(tf.truncated_normal([84,43], mean=mu, stddev=sigma))
    }

    biases = {
    'bc1': tf.Variable(tf.truncated_normal([10])),
    'bc2': tf.Variable(tf.truncated_normal([25])),
    'bd1': tf.Variable(tf.truncated_normal([120])),
    'bd2': tf.Variable(tf.truncated_normal([84])),
    'bd3': tf.Variable(tf.truncated_normal([43]))
    }
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x10.
    print(x)
    #input = tf.placeholder(tf.float32, shape=[None,32, 32, 1])
    conv_layer_1 = tf.nn.conv2d(x, weights['wc1'], strides=[1,1,1,1], padding='VALID')
    conv_layer_1 = tf.nn.bias_add(conv_layer_1, biases['bc1'])
    print(conv_layer_1)
    # TODO: Activation.
    
    conv_layer_1 = tf.nn.relu(conv_layer_1)

    # TODO: Pooling. Input = 28x28x10. Output = 14x14x10.
    
    pooling_1 = tf.nn.max_pool(conv_layer_1, [1,2,2,1], [1,2,2,1], padding='SAME')
    print(pooling_1)
    # TODO: Layer 2: Convolutional. Output = 10x10x25.
    
    conv_layer_2 = tf.nn.conv2d(pooling_1, weights['wc2'], strides=[1,1,1,1], padding='VALID')
    conv_layer_2 = tf.nn.bias_add(conv_layer_2, biases['bc2'])
    print(conv_layer_2)
    # TODO: Activation.
    
    conv_layer_2 = tf.nn.relu(conv_layer_2)

    # TODO: Pooling. Input = 10x10x25. Output = 5x5x25.
    
    pooling_2 = tf.nn.max_pool(conv_layer_2, [1,2,2,1], [1,2,2,1], padding='SAME')
    print(pooling_2)
    # TODO: Flatten. Input = 5x5x25. Output = 400.
    
    #fc1 = tf.reshape(pooling_2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = flatten(pooling_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    
    # TODO: Activation.
    
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    
    # TODO: Activation.
    
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    
    logits = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    EPOCHS_plt=[]
    validation_accuracy_plt=[]
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})    
        validation_accuracy = evaluate(X_validation, y_validation)
        
        validation_accuracy_plt.append(validation_accuracy) 
        EPOCHS_plt.append(i+1)
        
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    import numpy as np
    import matplotlib.pyplot as plt
    plt.plot(EPOCHS_plt,validation_accuracy_plt, 'ro')
    plt.axis([0, np.max(EPOCHS_plt), 0, 1])
    plt.title('Training Accuracy')
    plt.xlabel('EPOCHS')
    plt.ylabel('Validation Accuracy')
    plt.show()

    saver.save(sess, './traffic_v1')
    print("Model saved")






















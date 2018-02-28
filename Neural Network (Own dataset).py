# Acknowledgement:https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ#


import tensorflow as tf
import numpy as np


n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

n_classes = 4
batch_size = 100

# input feature size = 28x28 pixels = 784
x = tf.placeholder('float', [None, 15])
y = tf.placeholder('float')


def neural_network_model(data):
    # input_data * weights + biases
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([15, n_nodes_hl1])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_l2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_l3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}

    print(data.shape)
    print(hidden_l1['weights'].shape)
    print(hidden_l1['biases'].shape)

    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_l['weights']), output_l['biases'])
    return output


def train_neural_network(x,train_set_x,One_hot_matrix_train,test_set_x,One_hot_matrix_test): #,train_set_x,One_hot_matrix_train,test_set_x,One_hot_matrix_test
    print(x.shape)
    prediction = neural_network_model(x)
    print(prediction)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # v1.0 changes
    # optimizer value = 0.001, Adam similar to SGD
    print(cost.shape)
    print(cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs_no = 20001

    epoch_x = train_set_x.transpose()
    epoch_y = One_hot_matrix_train

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # v1.0 changes

        # training
        for epoch in range(epochs_no):
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss = c
            if(epoch%5000==0):
                print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss)

        # testing
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = accuracy.eval({x: test_set_x.transpose(), y: One_hot_matrix_test})
        print('Accuracy:', acc )


# train_set_x             ==>  (Number_of_Features * Number_of_training_data) Training matrix size
# One_hot_matrix_train    ==>  (Number_of_training_data * Number_of_class) Traing data
# test_set_x              ==>  (Number_of_Features * Number_of_test_data) Testing matrix size
# One_hot_matrix_test     ==>  (Number_of_test_data * Number_of_class) Test data


train_neural_network(variable,train_set_x,One_hot_matrix_train,test_set_x,One_hot_matrix_test)


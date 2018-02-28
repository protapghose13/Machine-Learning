import tensorflow as tf
import numpy as np
from Data_Processing_Sentiment_Analysis import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt','neg.txt',0.1)

n_nodes_hl1 = 600
n_nodes_hl2 = 600
n_nodes_hl3 = 600
n_nodes_hl4 = 600

n_classes = 2
batch_size = 100

# input feature size = 28x28 pixels = 784
x = tf.placeholder('float', [None, len(train_x[0]) ])
y = tf.placeholder('float')


def neural_network_model(data):
    # input_data * weights + biases
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_l2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_l3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_l4 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_l4['weights']), hidden_l4['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.add(tf.matmul(l4, output_l['weights']), output_l['biases'])
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # v1.0 changes
    # optimizer value = 0.001, Adam similar to SGD
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    print(np.array(train_x).shape)


    epochs_no = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # v1.0 changes

        for epoch in range(epochs_no):
            epoch_loss = 0
            i = 0
            while i<len(train_x):
                start = i
                end = i+batch_size
                i += batch_size
                epoch_x = np.array(train_x[start:end])
                epoch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss)

        # testing
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({ x: np.array(test_x) , y: np.array(test_y) } ))


train_neural_network(x)


import tensorflow as tf
import numpy as np

# Scale input
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

# Random
def shuffle_in_unison(t):
    a, b = t
    s = np.random.permutation(len(a))
    return a[s], b[s]

# For the softmax
def one_hot_encode(a):
    encoded = np.zeros((len(a), 10))
    for idx, val in enumerate(a):
        encoded[idx][val] = 1
    return encoded

def load_dataset():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Shuffle for random validation
    (x_train, y_train) = shuffle_in_unison((x_train, y_train))
    
    # Scale dataset (0-1)
    x_train, x_test = normalize(x_train), normalize(x_test)

    # One hot encoding
    y_train, y_test = one_hot_encode(y_train), one_hot_encode(y_test)
    
    # Actual split
    n_train = 49000
    return (x_train[:n_train],y_train[:n_train]), (x_train[n_train:],y_train[n_train:]), (x_test, y_test)  

def my_conv(x,keep_prob):
    
    A_conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    A_conv2 = tf.layers.conv2d(inputs=A_conv1, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    A_pool1 = tf.nn.dropout(tf.layers.max_pooling2d(inputs=A_conv2, pool_size=[2,2], strides=2), keep_prob)

    """A_conv1 = tf.layers.batch_normalization(A_pool1)"""

    A_conv3 = tf.layers.conv2d(inputs=A_pool1, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    A_conv4 = tf.layers.conv2d(inputs=A_conv3, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
    A_pool2 = tf.nn.dropout(tf.layers.max_pooling2d(inputs=A_conv4, pool_size=[2,2], strides=2), keep_prob)

    A_pool2_flat = tf.contrib.layers.flatten(A_pool2) 

    A_fc1 = tf.nn.dropout(tf.contrib.layers.fully_connected(inputs=A_pool2_flat, num_outputs=512, activation_fn=tf.nn.relu), keep_prob)
    """A_conv2 = tf.layers.batch_normalization(A_pool2)"""

    return tf.contrib.layers.fully_connected(inputs=A_fc1, num_outputs=10, activation_fn=None)

def main():

    np.random.seed(0xDEADBEEF)
    tf.random.set_seed(seed=0xDEADBEEF)

    train, valid, test = load_dataset()
    print("Lenght of the training set", len(train[0]), len(train[0]))
    print("Lenght of the validation set", len(valid[0]), len(valid[0]))
    print("Lenght of the test set", len(test[0]), len(test[0]))

    # Training procedure hyperparameters
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    keep_probability = 0.5

    tf.compat.v1.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Create model structure
    logits = my_conv(x,keep_prob)

    # Loss and Optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Accuracy
    hits = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(hits, tf.float32), name='accuracy')

    n_batches = len(train[0]) // batch_size
    print("batch_size", batch_size, "n_batches", n_batches)

    with tf.Session() as session:
        # Initializing the variables
        session.run(tf.global_variables_initializer())
        
        # Training cycle
        for epoch in range(epochs):

            # Shuffle training set to extract random validation set
            (x_train, y_train) = shuffle_in_unison(train)
            
            for i in range(n_batches):
                start = i*batch_size
                end = i*batch_size + batch_size
                """if(not i % (n_batches // 5)):
                    print("Progress ", start, ":", end)"""
                session.run(optimizer, feed_dict={x: x_train[start:end],y: y_train[start:end], keep_prob: keep_probability})

            l, a = session.run([loss, accuracy], feed_dict={x: valid[0],y: valid[1], keep_prob: 1.0}) #keep prob
            print(epoch+1, l, a)

        l, a = session.run([loss, accuracy], feed_dict={x: test[0],y: test[1], keep_prob: 1.0}) #keep prob
        print('Test', l, a)

# test
if __name__ == "__main__":
    main()
import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import trange

def logdir():
    now = datetime.utcnow().strftime("%Y-%m-%d%-H-%M-%S")
    root_logdir = "tf_dziennik"
    return "%s/przebieg-%s" % (root_logdir, now)

def get_batch(batch_index, batch_size):
    s = slice(batch_index * batch_size, (batch_index + 1) * batch_size)
    X_batch = X_train[s, :]
    y_batch = y_train[s]
    return X_batch, y_batch

FINAL_MODEL_PATH = './my_finished_model.ckpt'

if __name__ == "__main__":
    train()

def train():
    
    print("Tensorflow version:", tf.__version__)

    #load datasets and ceche them in ~/.keras/datasets/mnist.npz
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    # flatten each trainning and test example
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)


    # normalize train and test sets
    X_train = X_train / 256
    X_test = X_test / 256


    # neural network parameters
    n_examples, n_inputs = X_train.shape # 28 * 28 = 784
    n_epochs = 40
    batch_size = 100
    n_batches = int(np.ceil(n_examples / batch_size))

    learning_rate = 0.05
    momentum = 0.9

    n_hidden1 = 100
    n_hidden2 = 100
    n_outputs = 10

    with tf.name_scope("dnn"):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int64, shape=(None), name="y")
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="output")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("learning"):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        trainning_op = optimizer.minimize(loss)

    with tf.name_scope("evaluation"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("summary"):
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        loss_summary = tf.summary.scalar("loss", loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    final_model_path = FINAL_MODEL_PATH
    tmp_model_path = '/tmp/my_model.ckpt'

    with tf.Session() as sess, tf.summary.FileWriter(logdir(), tf.get_default_graph()) as file_writer:
        sess.run(init)
        for epoch in trange(n_epochs):
            batches_loss = 0
            for batch_index in range(n_batches):
                X_batch, y_batch = get_batch(batch_index, batch_size)
                _, loss_val = sess.run([trainning_op, loss], feed_dict={X:X_batch, y:y_batch})
                batches_loss += loss_val

            # file_writer.add_summary(acc_sum_res, epoch)
            # file_writer.add_summary(loss_sum_res, epoch)

            loss_val = loss.eval(feed_dict={X:X_train, y:y_train})
            acc_val = accuracy.eval(feed_dict={X:X_train, y:y_train})
            test_acc = accuracy.eval(feed_dict={X:X_test, y:y_test})
            print(f"Epoch {epoch}: Train loss {loss_val}, Train/Test accuracy {acc_val}/{test_acc}")

        print("Training finished. Test accuracy:", accuracy.eval(feed_dict={X:X_test, y:y_test}))

        saver.save(sess, final_model_path)
        print('Model saved at', final_model_path)



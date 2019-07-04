import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import trange
import matplotlib.pyplot as plt 
from scipy import ndimage

def logdir():
    now = datetime.utcnow().strftime("%Y-%m-%d%-H-%M-%S")
    root_logdir = "tf_dziennik"
    return "%s/przebieg-%s" % (root_logdir, now)

def get_batch(X, y, batch_index, batch_size):
    s = slice(batch_index * batch_size, (batch_index + 1) * batch_size)
    X_batch = X[s, :]
    y_batch = y[s]
    return X_batch, y_batch

FINAL_MODEL_PATH = './models/my_finished_model.ckpt'


def move_center_mass(X):
    # transorfms image so that it's center of mass is in the middle
    X = np.copy(X)
    for i in range(X.shape[0]):
        center = ndimage.measurements.center_of_mass(X[i])
        center_r = X.shape[1] / 2
        center_c = X.shape[2] / 2
        offset_r = int(np.round(center[0] - center_r))
        offset_c = int(np.round(center[1] - center_c))
        X_new = np.zeros(X[i, :, :].shape)
        #TODO: vectorize loops below
        for r in range(X.shape[1]):
            for c in range(X.shape[2]):
                if 0 <= r + offset_r < X.shape[1] and 0 <= c + offset_c < X.shape[2]:
                    X_new[r, c] = X[i, r + offset_r, c + offset_c]
        X[i] = X_new
    return X

class Model:

    def __init__(self):
        #load datasets and ceche them in ~/.keras/datasets/mnist.npz
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        self._data = {
            'train' : {
                'features' : X_train,
                'labels' : y_train
            },
            'test' : {
                'features' : X_test,
                'labels' : y_test
            }
        }
    
    def predict(self, x):
        x = np.array(x)
        
        x = move_center_mass(x)
        # plt.imshow(x[0].reshape(28, 28), cmap="gray")
        # plt.show()

        if np.any(x > 1):
            x = x / 255
        x = x.reshape(x.shape[0], -1)
        with tf.Session() as sess:    
            saver = tf.train.import_meta_graph(FINAL_MODEL_PATH + '.meta')
            saver.restore(sess, FINAL_MODEL_PATH)

            probabilities = sess.graph.get_tensor_by_name("dnn/probabilities:0")
            X = sess.graph.get_tensor_by_name("dnn/X:0")

            probabilities = probabilities.eval(feed_dict={X: x})

        prediction = np.argmax(probabilities, axis=1)[0]
        return prediction, probabilities

    def test_predictions(self):
        X_test = self._data['test']['features']
        y_test = self._data['test']['labels']

        X_test = X_test.reshape(X_test.shape[0], -1)
        X_test = X_test / 255

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(FINAL_MODEL_PATH + '.meta')
            saver.restore(sess, FINAL_MODEL_PATH)

            probabilities = sess.graph.get_tensor_by_name("dnn/probabilities:0")
            X = sess.graph.get_tensor_by_name("dnn/X:0")
            probabilities = probabilities.eval(feed_dict={X:X_test})
        i = 0
        while True:
            print("Showing next image. It's label is", y_test[i])
            plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
            plt.show()

            # X_moved = move_center_mass(np.array([X_test[i].reshape(28, 28)]))[0]
            # plt.imshow(X_moved, cmap="gray")
            # plt.show()

            i += 1

    def train(self):

        print("Tensorflow version:", tf.__version__)

        tf.reset_default_graph()


        X_train = self._data['train']['features']
        y_train = self._data['train']['labels']
        X_test = self._data['test']['features']
        y_test = self._data['test']['labels']

        # flatten each trainning and test example
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)


        # normalize train and test sets
        X_train = X_train / 256
        X_test = X_test / 256

        # neural network parameters
        n_examples, n_inputs = X_train.shape # 28 * 28 = 784
        
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
            probabilities = tf.nn.softmax(logits, name="probabilities")

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


        n_epochs = 40
        batch_size = 100
        n_batches = int(np.ceil(n_examples / batch_size))

        trainning_op = trainning_op
        loss = loss
        accuracy = accuracy
        accuracy_summary = accuracy_summary
        loss_summary = loss_summary


        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        saver = saver

        final_model_path = FINAL_MODEL_PATH
        tmp_model_path = '/tmp/my_model.ckpt'

        with tf.Session() as sess, tf.summary.FileWriter(logdir(), tf.get_default_graph()) as file_writer:
            sess.run(init)
            for epoch in trange(n_epochs):
                batches_loss = 0
                for batch_index in range(n_batches):
                    X_batch, y_batch = get_batch(X_train, y_train, batch_index, batch_size)
                    _, loss_val = sess.run([trainning_op, loss], feed_dict={X:X_batch, y:y_batch})
                    batches_loss += loss_val

                acc_sum_res, loss_sum_res = sess.run([accuracy_summary, loss_summary], feed_dict={X:X_train, y:y_train}) 
                file_writer.add_summary(acc_sum_res, epoch)
                file_writer.add_summary(loss_sum_res, epoch)

                loss_val = loss.eval(feed_dict={X:X_train, y:y_train})
                acc_val = accuracy.eval(feed_dict={X:X_train, y:y_train})
                test_acc = accuracy.eval(feed_dict={X:X_test, y:y_test})
                print(f"Epoch {epoch}: Train loss {loss_val}, Train/Test accuracy {acc_val}/{test_acc}")

            print("Training finished. Test accuracy:", accuracy.eval(feed_dict={X:X_test, y:y_test}))

            save__path = saver.save(sess, final_model_path)
            print('Model saved at', final_model_path)



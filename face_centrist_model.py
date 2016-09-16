import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from itertools import chain

class FaceCentrist():

    def __init__(self,num_layers,num_hidden,SEED=10100):

        # Parameters
        self.learning_rate = 0.001
        self.vector_size = 10
        self.grad_clip = 5.
        self.num_layers = num_layers

        # Network Parameters
        seq_max_len = 25  # Sequence max length
        self.n_hidden = num_hidden  # states of LSTMS
        n_classes = 1  # num of outputs (lin regression)


        # tf Graph input
        self.x = tf.placeholder("float", [None, seq_max_len, self.vector_size])
        self.y = tf.placeholder(tf.int64, [None, ])
        self.c = tf.placeholder("float", [None, 4064, ])
        # A placeholder for indicating each sequence length
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        # Define weights, centrist
        weights_cen = {
            'out': tf.Variable(tf.random_normal([4064, 512], seed=SEED))
        }
        biases_cen = {
            'out': tf.Variable(tf.random_normal([512], seed=SEED))
        }

        # Define weights, LSTMs
        weights_rnn = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, 16], seed=SEED))
        }
        biases_rnn = {
            'out': tf.Variable(tf.random_normal([16], seed=SEED))
        }


        # Define weights, output layer
        weights_out = {
            'out': tf.Variable(tf.random_normal([512 + 16, n_classes], seed=SEED))
        }
        biases_out = {
            'out': tf.Variable(tf.random_normal([n_classes], seed=SEED))
        }

        #Summaries for Tensorboard
        def variable_summaries(var, name):
            """Attach a lot of summaries to a Tensor."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.scalar_summary('mean/' + name, mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)

        #LSTMs
        def dynamicRNN(input, seqlen, weights, biases):

            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
            input = tf.transpose(input, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            input = tf.reshape(input, [-1, self.vector_size])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            input = tf.split(0, seq_max_len, input)
            #Summary of variables
            variable_summaries(input, 'inputs')
            variable_summaries(biases['out'], 'biases')
            variable_summaries(weights['out'], 'weights')


            # Define a lstm cell with tensorflow
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            if num_layers > 1:
                rnn = rnn_cell.MultiRNNCell([cell] * num_layers)
            else:
                rnn = cell

            # Get lstm cell output,
            outputs, states = tf.nn.rnn(rnn, input, dtype=tf.float32,
                                        sequence_length=seqlen)

            # 'outputs' is a list of output at every timestep, we pack them in a Tensor
            # and change back dimension to [batch_size, n_step, n_input]
            outputs = tf.pack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])

            batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
            # Indexing
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)

            # Linear activation, using outputs computed above
            return tf.matmul(outputs, weights['out']) + biases['out']

        # Define CENTRIST Layer
        def centrist_layer(x, weights, biases):

            preactivate = tf.matmul(x, weights['out']) + biases['out']
            hidden = tf.nn.relu(preactivate)
            tf.histogram_summary('hidden_centrist/pre_activations', preactivate)
            tf.histogram_summary('hidden_centrist/activations', hidden)
            hidden = tf.nn.dropout(hidden, self.dropout_keep_prob)

            # Linear activation, using outputs computed above
            return hidden

        rnn_output = dynamicRNN(self.x, self.seqlen, weights_rnn, biases_rnn)
        centrist_output = centrist_layer(self.c, weights_cen, biases_cen)

        #Concatanate output
        combo = tf.concat(1, [rnn_output, centrist_output])

        self.pred = tf.matmul(combo, weights_out['out']) + biases_out['out']

        y_f = tf.cast(self.y, tf.float32)
        y_pred = tf.reshape(self.pred, [tf.shape(self.y)[0], ])

        # Mean squared error
        batch_size = tf.cast(tf.shape(self.y)[0],tf.float32)
        self.cost = tf.reduce_sum(tf.pow(y_pred - y_f, 2)) / (2 * batch_size)

        tf.scalar_summary('loss', self.cost)
        # Compute RMSE
        self.rmse = tf.sqrt(

            tf.reduce_sum(tf.pow(tf.reshape(self.pred, [tf.shape(self.y)[0], ]) - y_f, 2) / tf.cast(tf.shape(self.y), tf.float32)))
        # Evaluate model
        tf.scalar_summary('rmse', self.rmse)
        self.merged_summaries = tf.merge_all_summaries()

        # do gradient clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          self.grad_clip)

        self.lr = tf.Variable(self.learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def infer(self, sess, face_features, seq_len, centrist):

        # Feed it to the graph, fetch predictions
        predictions = sess.run([self.pred], feed_dict={self.x: face_features,
                                                                           self.seqlen: seq_len,
                                                                           self.dropout_keep_prob: 1, self.c: centrist})

        predictions = list(chain.from_iterable(predictions))
        return predictions

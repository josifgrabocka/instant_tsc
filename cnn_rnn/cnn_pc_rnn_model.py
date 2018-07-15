import tensorflow as tf
import numpy as np
import scipy.stats
from cnn_rnn.smoothing_coefficients import SmoothingCoeffs


class CNN_PC_RNN_Model:

    def __init__(self, dataset, config, verbose=False):

        # reset the graph for a new execution
        tf.reset_default_graph()

        self.config = config

        self.rnn_layer_sizes = self.config['cnn_rnn:rnn_layer_sizes']
        self.cnn_layer_sizes = self.config['cnn_rnn:cnn_layer_sizes']
        self.nn_layer_sizes = self.config['cnn_rnn:nn_layer_sizes']
        self.data = dataset
        self.mode = self.config['cnn_rnn:mode']
        self.demanded_frac = self.config['cnn_rnn:demanded_frac']
        self.max_pool_size = config['cnn_rnn:max_pool_size']

        self.verbose = verbose

        if self.verbose == False:
            print('Mode=', self.mode, ', Demanded fraction=', self.demanded_frac)

        # placeholder for the series batch as (batch_size, num_segments, segment_length)
        self.X_batch = tf.placeholder(shape=[None, self.data.series_length, self.data.num_channels], dtype=tf.float32)
        # a flag to denote if it is in training mode, used for batch normalization
        self.is_training = tf.placeholder(tf.bool)
        # the dropout rates
        self.drop_rate = tf.placeholder(tf.float32)
        self.one_tensor = tf.constant(1.0, dtype=tf.float32)
        # create the prediction model
        self.create_model()


    # create the prediction model
    def create_model(self):
        # create the cnn rnn model
        self.create_cnn_rnn_model()
        # finally aggregate the predictons
        self.aggregate_predictions()

    # create a mixture of cnn and rnn model
    def create_cnn_rnn_model(self):

        with tf.name_scope("InputLayers"):
            self.conv_output = tf.layers.batch_normalization(self.X_batch, training=self.is_training)

        with tf.name_scope("CNNLayers"):
            self.num_segments = self.data.series_length

            for num_neurons in self.cnn_layer_sizes:
                # kernel length 5% of the length of previous feature map
                kernel_length = np.int(np.ceil(self.num_segments * self.config['cnn_rnn:kernel_length_frac']))

                # let the stride be % of the kernel length, i.e. 1-% of the kernel span is shared in consecutive convolutions
                self.conv_output = tf.layers.conv1d(inputs=self.conv_output, filters=num_neurons,
                                                        padding='SAME',
                                                        kernel_size=kernel_length)
                # add a max normalzation after the convolution
                self.conv_output = tf.layers.batch_normalization(self.conv_output, training=self.is_training)
                # nonlinear activation
                self.conv_output = tf.nn.relu(self.conv_output)
                # read the number of segments after the convolution
                self.num_segments = self.conv_output.get_shape().as_list()[1]
                # print info about the convolution
                if self.verbose == False:
                    print('Conv layer, num_kernels', num_neurons, ', kernel_length', kernel_length, 'altered dims to ', self.conv_output.shape)

            # add a pooling layer before the rnn to avoid large lengths
            self.conv_output = tf.layers.max_pooling1d(inputs=self.conv_output, pool_size=self.max_pool_size,
                                                   strides=self.max_pool_size)
            self.num_segments = self.conv_output.get_shape().as_list()[1]

            if self.verbose == False:
                print('Pooling layer, pool_size', self.max_pool_size, 'altered dims to ', self.conv_output.shape)

        with tf.name_scope("RNNLayers"):

            last_num_channels = self.conv_output.shape[2]

            num_cells_per_channel = self.config['cnn_rnn:rnn_cells_per_channel']
            rnn_activations_per_channel = []

            # create the multi layer rnn cells
            for channel_idx in range(last_num_channels):

                # slice the channel and expand to keep the last dim as 1
                self.conv_output_channel = tf.expand_dims(self.conv_output[:, :, channel_idx], -1)
                # transpose the tensor to (num_segments, batch_size, num kernels)
                self.conv_output_channel = tf.transpose(self.conv_output_channel, perm=[1, 0, 2])
                # unstack to a list of num_segments tensors of shape (batch_size, segment_length)
                self.conv_output_channel_list = tf.unstack(self.conv_output_channel, axis=0)

                # define rnn and get activations
                self.segments_activations_list, states_list = \
                    tf.nn.static_rnn(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=num_cells_per_channel,
                                                                 activation=tf.nn.relu,
                                                                 name="channel"+str(channel_idx)),
                                     inputs=self.conv_output_channel_list,
                                     dtype=tf.float32)
                # stack the activations
                channel_activations = tf.stack(self.segments_activations_list)

                # the rnn activations of the cells for this channel are grouped into
                rnn_activations_per_channel.append(channel_activations)

                # add batch normalizations
                print('RNN: ', num_cells_per_channel, 'cells for channel', channel_idx, 'shape', channel_activations.shape)

                    #rnn_cells.clear()

            # stack the activations of the rnns of each channel
            stacked_activations =  tf.stack(rnn_activations_per_channel, axis=2)
            # merge the activations of all the channels, last two dims
            self.segments_activations = tf.reshape(tensor=stacked_activations, shape=(self.conv_output.shape[1],
                                                                                      -1,
                                                                                      self.conv_output.shape[2]*num_cells_per_channel))
            # batch normalize activations
            self.segments_activations = tf.layers.batch_normalization(self.segments_activations, training=self.is_training)


            # keep these for compatibility purposes with other variants
            # transpose the tensor to (num_segments, batch_size, num kernels)
            self.conv_output = tf.transpose(self.conv_output, perm=[1, 0, 2])
            # unstack to a list of num_segments tensors of shape (batch_size, segment_length)
            self.conv_output_list = tf.unstack(self.conv_output, axis=0)

            if self.verbose == False:
                print('RNN dims', self.segments_activations.shape)

    # aggregate the predictions
    def aggregate_predictions(self):

        with tf.name_scope("PredictionLayers"):

            # add fully connected layers
            self.predictions = self.segments_activations
            for num_neurons in self.nn_layer_sizes:
                print('NN layer, num_neurons', num_neurons)
                # add a dense layer for the predictions
                self.predictions = tf.layers.dense(inputs=self.predictions, units=num_neurons)
                self.predictions = tf.layers.batch_normalization(self.predictions, training=self.is_training)
                self.predictions = tf.nn.relu(self.predictions)

            # and the last logits predictions
            self.predictions = tf.layers.dense(inputs=self.predictions, units=self.data.num_classes)

            # aggregate the predictions to compute the final prediction
            if self.mode == 'last':
                self.final_prediction = self.predictions[-1]
            elif self.mode == 'mean':
                self.final_prediction = tf.reduce_mean(self.predictions, axis=0)
            elif self.mode == 'max':
                self.final_prediction = tf.reduce_max(self.predictions, axis=0)
            # the exponential smoothing
            elif self.mode == 'exp' or self.mode == 'multitask':

                demanded_index = int(self.demanded_frac*self.num_segments)

                # compute the multi-task weights
                if self.config['cnn_rnn:smoothing_type'] == 'decay':
                    self.smoco = SmoothingCoeffs(T=self.num_segments,
                                            T_target=demanded_index,
                                            percentile=self.config['cnn_rnn:smoothing_percentile'])
                    coefficients, _, _ = self.smoco.solve()
                    # make the coefficients a column vector, useful for the multiplication below
                    coefficients_np = np.array([[xi] for xi in coefficients], dtype=np.float32)
                    # make a tensor from the coefficients vector and add a dummy dimension to (num segments x 1)
                    self.smoothing_coeffs = tf.expand_dims(tf.convert_to_tensor(coefficients_np), axis=-1)
                
                # give importance to coefficients around the demanded index and decay nearby
                elif self.config['cnn_rnn:smoothing_type'] == 'norm':
                    x = np.arange(0, self.num_segments, 1)
                    scale = self.config['cnn_rnn:norm_var'] * self.num_segments
                    coeffs = scipy.stats.norm.pdf(x, loc=demanded_index, scale=scale).astype(np.float32)
                    coeffs /= np.sum(coeffs)
                    self.smoothing_coeffs = tf.expand_dims(tf.convert_to_tensor(coeffs), axis=-1)

                if self.mode == 'exp':
                    self.final_prediction = tf.reduce_sum(tf.multiply(self.predictions, self.smoothing_coeffs), axis=0)

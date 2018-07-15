import tensorflow as tf
import numpy as np
import math
import scipy.stats
from cnn_rnn.smoothing_coefficients import SmoothingCoeffs


class CNN_Model:

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

        self.feature_maps = []

        with tf.name_scope("CNNLayers"):
            self.num_segments = self.data.series_length
            self.conv_output = self.X_batch

            for num_filters in self.cnn_layer_sizes:

                kernel_length = max(1, int(self.num_segments * self.config['cnn_rnn:kernel_length_frac']))

                self.conv_output = tf.layers.conv1d(inputs=self.conv_output, filters=num_filters,
                                                            padding='SAME',
                                                            kernel_size=kernel_length)
                self.conv_output = tf.layers.batch_normalization(self.conv_output, training=self.is_training)
                self.conv_output = tf.nn.relu(self.conv_output)
                self.num_segments = self.conv_output.get_shape().as_list()[1]

                if self.verbose == False:
                    print('CNN layer dim ', self.conv_output.shape, ', kernel_length ', kernel_length)
                self.feature_maps.append(self.conv_output)

        with tf.name_scope("AggregateActivationsLayer"):
            # transpose the tensor to (num_segments, batch_size, num kernels)
            self.conv_output = tf.transpose(self.conv_output, perm=[1, 0, 2])
            # unstack to a list of num_segments tensors of shape (batch_size, segment_length)
            self.conv_output_list = tf.unstack(self.conv_output, axis=0)

            # the cumulative average across the series activations
            self.numerical_progression = tf.cumsum(tf.ones_like(self.conv_output), axis=0)
            self.segments_activations = tf.div(tf.cumsum(self.conv_output, axis=0), self.numerical_progression)


    # aggregate the predictions
    def aggregate_predictions(self):

        # and the last logits predictions
        self.predictions = tf.layers.dense(inputs=self.segments_activations, units=self.data.num_classes)

        # reduce resolution
        demanded_index = int(self.demanded_frac * self.num_segments)

        if self.config['cnn_rnn:restrict_resolution'] == True:
            kernel_length = max(1, int(self.data.series_length * self.config['cnn_rnn:kernel_length_frac']))
            num_layers = len(self.config['cnn_rnn:cnn_layer_sizes'])
            resolution = int((kernel_length/2))* num_layers
            print('D changed from', demanded_index, 'back by', resolution)
            # fix resolution
            demanded_index -= resolution
            if demanded_index < 0:
                demanded_index = 0

        # aggregate the predictions to compute the final prediction
        if self.mode == 'last':
            self.final_prediction = self.predictions[-1]
        elif self.mode == 'mean':
            self.final_prediction = tf.reduce_mean(self.predictions, axis=0)
        elif self.mode == 'max':
            self.final_prediction = tf.reduce_max(self.predictions, axis=0)
        # the exponential smoothing
        elif self.mode == 'exp' or self.mode == 'multitask':


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

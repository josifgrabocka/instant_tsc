import numpy as np
import tensorflow as tf


class SmoothingCoeffs:

    def __init__(self, T, T_target, percentile,
                 num_epochs=20000,
                 eta=0.001):

        self.eta = eta
        self.num_epochs = num_epochs

        print(percentile, T_target)

        # avoid T_target == 0 
        if T_target == 0:
            T_target = 1

        with tf.name_scope('SmoothingCoefficients'):
            self.alpha = tf.Variable(initial_value=tf.zeros(shape=[]), dtype=tf.float32)
            self.beta = tf.Variable(initial_value=tf.zeros(shape=[]), dtype=tf.float32)

            self.T = tf.constant(T, dtype=tf.float32)
            self.T_target = tf.constant(T_target, dtype=tf.float32)

            one_tensor = tf.constant(1.0, dtype=tf.float32)
            percentile_tensor = tf.constant(percentile, dtype=tf.float32)

            # define the tensor for the coefficients
            self.indices_range = tf.range(start=0, limit=self.T, delta=1, dtype=tf.float32)
            self.norm_indices = tf.map_fn(lambda t: t / self.T, self.indices_range, dtype=tf.float32)
            self.coeffs = tf.map_fn(lambda t: tf.multiply(self.alpha, tf.pow((one_tensor-t),self.beta)), self.norm_indices, dtype=tf.float32)

            self.sum_coeffs = tf.reduce_sum(self.coeffs)
            self.targeted_coeffs = tf.slice(self.coeffs, begin=(0,), size=(T_target,))
            self.sum_target_coeffs = tf.reduce_sum(self.targeted_coeffs, axis=0)

            self.loss = tf.squared_difference(one_tensor, self.sum_coeffs) \
                        + tf.squared_difference(percentile_tensor, self.sum_target_coeffs)

            # define an update step for gradient descent using the adam optimizer
            self.update_step = tf.train.AdamOptimizer(self.eta).minimize(self.loss)

    # solve the parameters of the smoothing coefficients
    def solve(self):

        coeffs_val, alpha_val, beta_val = None, None, None

        #update the alpha and beta varibles to minimize the loss

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config) as sess:
            # initialize variables alpha and beta
            sess.run(tf.global_variables_initializer())

            # optimize the loss for a series of epochs
            for iter in range(self.num_epochs):
                _, loss_val, alpha_val, beta_val, sum_val, sum_target_val = sess.run([self.update_step, self.loss,
                                                                                      self.alpha, self.beta,
                                                                                      self.sum_coeffs, self.sum_target_coeffs])
                # print the progres logs
                if iter % 1000 == 0:
                    print('Iter', iter, 'Loss', loss_val, 'Alpha', alpha_val, 'Beta', beta_val,
                          'Sum', sum_val, 'SumTarget', sum_target_val)

                    #print( sess.run([self.targeted_coeffs, self.sum_target_coeffs]) )

            # final value of coefficients and smoothing parameters
            coeffs_val, alpha_val, beta_val = sess.run([self.coeffs, self.alpha, self.beta])

        # value of coefficients divided by sum to make them perfectly sum to one
        coeffs_val /= np.sum(coeffs_val)

        return coeffs_val, alpha_val, beta_val



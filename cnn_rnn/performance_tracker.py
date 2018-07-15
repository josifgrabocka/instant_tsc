import tensorflow as tf
import os
import numpy as np

# this class collect summaries, logs, and other performance indicators
class PerformanceTracker:

    def __init__(self, optimizer, dataset, model, config):

        self.optimizer = optimizer
        self.dataset = dataset
        self.model = model
        self.config = config

        self.summary_logs_folder = "summary_logs/" + dataset.dataset_name + "_" + model.mode
        # remove folder if it exists
        if os.path.exists(self.summary_logs_folder):
            os.system("rm -rf " + self.summary_logs_folder)

        # add a summary of the unregularized loss
        tf.summary.scalar("accuracies/loss", self.optimizer.unregularized_loss)

        # define the early accuracies
        self.define_accuracies()

        # define the gradients wrt conv
        self.define_gradients_wrt_conv()

        # an op that merges summaries
        self.summary_merger = tf.summary.merge_all()

        # create a saver
        self.saver = tf.train.Saver()

        # set if the performance tracker should be turned on or off
        self.verbose = True

    # updates before the gradient descent opt steps
    def pre_opt_steps_update(self, sess):
        if self.verbose == False:
            self.summary_writer = tf.summary.FileWriter(self.summary_logs_folder, sess.graph)

    # updates being run during each iteration update
    def opt_steps_update(self, sess, epoch):
        if self.verbose == False:
            # collect summaries
            summary_vals = sess.run(self.summary_merger, feed_dict={self.model.X_batch: self.dataset.X_test,
                                                                       self.optimizer.Y_batch: self.dataset.Y_test,
                                                                       self.model.is_training: False,
                                                                       self.model.drop_rate: 1.0})

            #write them to file
            self.summary_writer.add_summary(summary_vals, global_step=epoch)


    # updates being run when there is an improvement in the training loss
    def save_model(self, sess):
        self.saver.save(save_path='models/' + self.dataset.dataset_name + '.ckp', sess=sess)

    # measure the early accuracies on the test set
    def compute_test_accuracy(self, sess):

        num_test_instances = self.dataset.X_test.shape[0]
        total_accuracy = []
        num_batches = 0
        batch_size = self.config['performancetracker:batch_size']

        # iterate through a series of batches
        for idx in range(0, num_test_instances, batch_size):

            # get the batch accuracy
            X_b, Y_b = self.dataset.X_test[idx:idx + batch_size], self.dataset.Y_test[idx:idx + batch_size]
            batch_accuracy = self.compute_accuracies(sess, X_b, Y_b)

            # initialize the batch accuracy vector for the first time
            if idx == 0:
                total_accuracy = batch_accuracy
            # sum up the accuracies
            else:
                for earliness_idx, val in enumerate(batch_accuracy):
                    total_accuracy[earliness_idx] += batch_accuracy[earliness_idx]

            num_batches += 1

        # compute mean from the sum of accuracies
        for earliness_idx, val in enumerate(total_accuracy):
            total_accuracy[earliness_idx] /= num_batches

        return total_accuracy

    # compute the value of the training loss
    # compute it in batches for very large datasets
    def compute_training_loss(self, sess):

        num_train_instances = self.dataset.X_train.shape[0]
        total_loss = 0
        num_batches = 0
        batch_size = self.config['performancetracker:batch_size']

        for idx in range(0, num_train_instances, batch_size):
            X_b, Y_b = self.dataset.X_train[idx:idx+batch_size], self.dataset.Y_train[idx:idx+batch_size]
            total_loss += self.compute_loss(sess, X_b, Y_b)
            num_batches += 1

        return total_loss / num_batches

    # compute the loss of a batch
    def compute_loss(self, sess, X_b, Y_b):
        return sess.run(self.optimizer.loss, feed_dict={self.model.X_batch: X_b,
                                                        self.optimizer.Y_batch: Y_b,
                                                        self.model.is_training: False,
                                                        self.model.drop_rate: 0.0})

    # compute the accuracy of a batch
    def compute_accuracies(self, sess, X_b, Y_b):
        return sess.run(self.early_accuracies, feed_dict={self.model.X_batch: X_b,
                                                          self.optimizer.Y_batch: Y_b,
                                                          self.model.is_training: False,
                                                          self.model.drop_rate: 0.0})

    # define the gradients
    def define_gradients_wrt_conv(self):

        #
        # # for each percentage level
        # for frac in np.arange(start=0.1, stop=0.91, step=0.1):
        #     # get the prediction for the percentage of series
        #     demanded_idx = np.int(np.ceil(self.model.num_segments * frac))
        #
        #     # avoid that the index is outside the allowed range
        #     if demanded_idx == self.model.num_segments:
        #         demanded_idx = self.model.num_segments - 1
        #
        #     # get the average gradient of the loss wrt to the convolution feature map at the demanded index
        #     norm_grads_conv_demanded_idx = tf.norm(tf.gradients(self.optimizer.loss,
        #                                                           self.model.conv_output_list[demanded_idx])[0])
        #
        #     tf.summary.scalar('gradients/convolutions/c_'+str(frac), norm_grads_conv_demanded_idx)
        #
        #     # get the average gradient of the loss wrt to the convolution feature map at the demanded index
        #     norm_grads_segs_act_demanded_idx = tf.norm(tf.gradients(self.optimizer.loss,
        #                                                                self.model.segments_activations_list[demanded_idx])[0])
        #
        #     tf.summary.scalar('gradients/activations/a_' + str(frac), norm_grads_segs_act_demanded_idx)
        #

        self.grads_feature_maps = []

        self.grads_feature_maps.append( tf.gradients(self.optimizer.loss, self.model.X_batch)[0] )

        for l in range(len(self.config['cnn_rnn:cnn_layer_sizes'])):
            self.grads_feature_maps.append(tf.gradients(self.optimizer.loss, self.model.feature_maps[l])[0])


    # define the early classification accuracies
    def define_accuracies(self):
        # define the final classification accuracy
        correct_predictions = tf.equal(tf.argmax(self.optimizer.Y_batch, 1), tf.argmax(self.model.predictions[-1], 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # add the final accuracy as a the summary
        tf.summary.scalar("accuracies/acc", self.accuracy)

        # define early classification accuracies
        self.early_length_fractions = np.arange(start=0.05, stop=1.01, step=0.05) 
        self.early_accuracies = []
        for frac in self.early_length_fractions:
            # get the prediction for the percentage of series
            demanded_idx = np.int(np.ceil(self.model.num_segments * frac))

            # avoid that the index is outside the allowed range
            if demanded_idx == self.model.num_segments:
                demanded_idx = self.model.num_segments - 1

            # define the classification accuracy
            correct_predictions = tf.equal(tf.argmax(self.optimizer.Y_batch, 1), tf.argmax(self.model.predictions[demanded_idx], 1))
            self.early_accuracies.append(tf.reduce_mean(tf.cast(correct_predictions, tf.float32)))

        # add histogram summaries for all the early accuracies
        for idx, frac in enumerate(self.early_length_fractions):
            tf.summary.scalar("accuracies/early_"+str(frac), self.early_accuracies[idx])


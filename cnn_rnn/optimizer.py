import tensorflow as tf
import sys

from cnn_rnn.performance_tracker import PerformanceTracker

class Optimizer:

    def __init__(self, config):

        self.config = config

        self.num_epochs = self.config['optim:num_epochs']
        self.batch_size = self.config['optim:batch_size']
        self.features = None
        self.best_model = ""
        self.eta = self.config['optim:eta']
        self.drop_rate = self.config['optim:drop_rate']
        self.max_grad_norm = self.config['optim:max_grad_norm']
        self.best_early_results = []
        self.tolerance_divergence_steps = self.config['optim:tolerance_divergence_steps']
        self.early_accuracies = None
        self.close_session = True
        self.sess = None

    # optimize the early classification of time series
    def optimize(self, dataset, model, run_initializer=True):

        # a counter for the current epoch
        epoch = 1

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        # open session
        self.sess = tf.Session(config=config)

        # initialize all variables
        if run_initializer == True:
            self.sess.run(tf.global_variables_initializer())

        # run the pre-opt performance tracker
        self.perf_trac.pre_opt_steps_update(sess=self.sess)

        # repeat in a number of iterations
        smallest_loss = 10000.0

        last_improvement_epoch = 0

        # run the performance tracker before the opt steps, e.g. to create summary file writer
        self.perf_trac.opt_steps_update(sess=self.sess, epoch=epoch)

        while True:

            # check for convergence and tolerate a couple of steps
            if epoch > self.num_epochs:
                break

            self.single_epoch_optimize(self.sess, dataset, model)

            # at the beginning of an epoch log results
            # compute the training loss and check if there is an improvement
            if epoch % 100 == 0: 
                train_loss_val = self.perf_trac.compute_training_loss(sess=self.sess)

                if train_loss_val < smallest_loss:
                    # record the best accuracies so far
                    self.best_early_results = self.perf_trac.compute_test_accuracy(sess=self.sess)

                    # print the loss and early accuracies
                    if not self.config['optim:verbose']:
                        print(epoch, dataset.dataset_name, model.mode, ', TrLo=', train_loss_val, ', TeAc=', self.best_early_results)
                        sys.stdout.flush()

                    # store the smallest loss so far
                    smallest_loss = train_loss_val
                    last_improvement_epoch = epoch

                    # run the final step of the performance tracker
                    #self.perf_trac.save_model(sess=sess)

            # increment the batch id
            epoch += 1

        # close session
        if self.close_session:
            self.sess.close()

    # optimize the early classification of time series
    def single_epoch_optimize(self, open_sess, dataset, model):

        # draw a random batch from this task
        X_b, Y_b = dataset.draw_batch(self.batch_size)
        # run one update step on this all the variables of the prediction model
        open_sess.run(self.update_step, feed_dict={model.X_batch: X_b,
                                                   self.Y_batch: Y_b,
                                                   model.is_training: True,
                                                   model.drop_rate: self.drop_rate})


    # define the losses of the model
    def define_losses(self, dataset, model):

        # the tensors for the loss and update step
        with tf.name_scope("Performance"):

            # a placeholder for the labels of the batch
            self.Y_batch = tf.placeholder(dtype=tf.float32, shape=[None, dataset.num_classes])

            if model.mode == 'multitask':
                # define the loss
                indices_range = tf.range(start=0, limit=model.num_segments, delta=1, dtype=tf.int32)

                self.time_losses =  tf.map_fn(lambda t: tf.multiply(model.smoothing_coeffs[t],
                                                                    tf.reduce_mean(
                                                                        tf.nn.softmax_cross_entropy_with_logits(
                                                                            labels=self.Y_batch,
                                                                            logits=model.predictions[t]))),
                                              indices_range,
                                              dtype=tf.float32)

                self.unregularized_loss = tf.reduce_sum(self.time_losses)
            elif model.mode == 'singletask':
                # define the loss
                demanded_idx = int(model.demanded_frac * model.num_segments)

                if demanded_idx >= model.num_segments:
                    demanded_idx = model.num_segments - 1

                self.unregularized_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_batch, logits=model.predictions[demanded_idx]))
            else:
                self.unregularized_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_batch, logits=model.final_prediction))

            # add L2 regularization to the loss
            trainable_vars = tf.trainable_variables()
            self.reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars]) * self.config['optim:lambda']
            self.loss = self.unregularized_loss + self.reg_loss

            # apply the gradients using clipping to avoid their explosion
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), self.max_grad_norm)
                optimizer = tf.train.AdamOptimizer(self.eta)
                gradients = zip(clipped_grads, trainable_vars)
                self.update_step = optimizer.apply_gradients(gradients)

            # meawhile make sure that batch size does not exceed the number of instances in toy datasets
            self.batch_size = self.config['optim:batch_size']

            # pring the hyper-parameters of the model
            print('Mini-batch size', self.batch_size, 'Learning rate', self.eta, 'Max gradient norm', self.max_grad_norm,
                  'Epochs', self.num_epochs, 'Tolerance steps', self.tolerance_divergence_steps)

        # initialize a performance tracker
        self.perf_trac = PerformanceTracker(optimizer=self, dataset=dataset, model=model, config=self.config)

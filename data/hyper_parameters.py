import numpy as np


class HyperParams:

    def __init__(self):

        self.type = 'univariate'
        pass

    # get the setup
    def get_config(self, data, argv):

        self.data = data

        if self.type == 'univariate':
            K = int(np.log(data.num_train_instances) * np.log(data.num_classes))
            return self.early_config(K, argv)
        elif self.type == 'multivariate':
            K = int(np.log(data.num_train_instances)*np.log(data.num_classes))
            return self.multivariate_config(K, argv)

    def early_config(self, K, argv):

        config = {}
        config['cnn_rnn:cnn_layer_sizes'] = [64,64,64,64,64]
        config['cnn_rnn:rnn_layer_sizes'] = []
        config['cnn_rnn:nn_layer_sizes'] = []
        config['cnn_rnn:rnn_cells_per_channel'] = 1
        config['cnn_rnn:init_var'] = 0.001
        config['cnn_rnn:mode'] = argv[2]
        config['cnn_rnn:demanded_frac'] = float(argv[3])
        config['cnn_rnn:kernel_length_frac'] = 0.075
        config['cnn_rnn:max_pool_size'] = 2

        if config['cnn_rnn:mode'] == 'multitask':
            config['cnn_rnn:smoothing_type'] = argv[4]
            config['cnn_rnn:smoothing_percentile'] = 0.5
            config['cnn_rnn:norm_var'] = 0.1

        config['optim:num_epochs'] = 1000
        config['optim:batch_size'] = 10
        config['optim:eta'] = 0.001
        config['optim:drop_rate'] = 0.0
        config['optim:max_grad_norm'] = 100.0
        config['optim:lambda'] = 0.0
        config['optim:verbose'] = False
        config['optim:tolerance_divergence_steps'] = 10
        config['performancetracker:batch_size'] = 1

        return config


    def multivariate_config(self, K, argv):

        config = {}
        config['cnn_rnn:cnn_layer_sizes'] = [64, 64, 64, 64]
        config['cnn_rnn:rnn_layer_sizes'] = []
        config['cnn_rnn:nn_layer_sizes'] = []
        config['cnn_rnn:rnn_cells_per_channel'] = 1
        config['cnn_rnn:init_var'] = 0.001
        config['cnn_rnn:mode'] = argv[2]
        config['cnn_rnn:demanded_frac'] = float(argv[3])
        config['cnn_rnn:kernel_length_frac'] = 0.05
        config['cnn_rnn:restrict_resolution'] = True

        if config['cnn_rnn:mode'] == 'multitask':
            config['cnn_rnn:smoothing_type'] = argv[4]
            config['cnn_rnn:smoothing_percentile'] = 0.3+float(argv[3])
            config['cnn_rnn:norm_var'] = 0.15

        config['optim:num_epochs'] = 100000
        config['optim:batch_size'] = 100
        config['optim:eta'] = 0.001
        config['optim:max_grad_norm'] = 10.0
        config['optim:drop_rate'] = 0.0
        config['optim:lambda'] = 0.0
        config['optim:verbose'] = False
        config['optim:tolerance_divergence_steps'] = 10
        config['performancetracker:batch_size'] = 1

        return config

    def multitask_config(self, K, argv):

        config = {}
        config['cnn:num_common_channels'] = [128, 64]
        config['cnn:common_length'] = [128, 64]

        config['cnn:cnn_layer_sizes'] = []
        config['cnn:cnn_layer_sizes'] = []

        config['cnn_rnn:init_var'] = 0.001
        config['cnn_rnn:mode'] = argv[2]
        config['cnn_rnn:demanded_frac'] = float(argv[3])
        config['cnn_rnn:kernel_length_frac'] = 0.1
        config['cnn_rnn:kernel_stride_frac'] = 0.2

        if config['cnn_rnn:mode'] == 'multitask':
            config['cnn_rnn:smoothing_type'] = argv[4]
            config['cnn_rnn:smoothing_percentile'] = 0.5
            config['cnn_rnn:norm_var'] = 0.2

        config['optim:num_epochs'] = 1000
        config['optim:batch_size'] = np.min([100, np.max([5, self.data.num_train_instances // 20])])
        config['optim:eta'] = 0.0003
        config['optim:drop_rate'] = 0.5
        config['optim:max_grad_norm'] = 1
        config['optim:verbose'] = False
        config['optim:tolerance_divergence_steps'] = 50
        config['performancetracker:batch_size'] = 1

        return config

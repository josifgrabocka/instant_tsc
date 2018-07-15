from data.dataset import Dataset
from data.hyper_parameters import HyperParams
#from cnn_rnn.cnn_rnn_model import CNN_RNN_Model
from cnn_rnn.cnn_rnn_model import CNN_RNN_Model
from cnn_rnn.optimizer import Optimizer
import tensorflow as tf
import sys
import time

# load the data
dataset_path=sys.argv[1]
data = Dataset()
#data.load_multivariate(dataset_prefix=dataset_path, train_test_fraction=0.9)
data.load_ucr_univariate_data(dataset_folder=dataset_path)

#start measuring time
start_time = time.time()

# get the hyper parameters for speech
hyperparams = HyperParams()
hyperparams.type = 'univariate'
config = hyperparams.get_config(data, sys.argv)

# mask the test points after the demanded fraction
data.mask_test_after(config['cnn_rnn:demanded_frac'])

# reset the graph for a new execution
tf.reset_default_graph()

# create final model with the best smoothing degree
model = CNN_RNN_Model(dataset=data, config=config)

# create a list of tasks
opt = Optimizer(config=config)
opt.define_losses(dataset=data, model=model)
opt.optimize(dataset=data, model=model)

# stop measuring time
train_time = time.time() - start_time

# just a tweak:
# add the smoothing mode suffix to the method name, before printing
if sys.argv[2] == 'multitask':
    sys.argv[2] = sys.argv[2] + '_' +  sys.argv[4]

# print the results
print(data.dataset_name, train_time, sys.argv[2:4], opt.best_early_results)

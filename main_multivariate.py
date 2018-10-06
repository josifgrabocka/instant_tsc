from data.dataset import Dataset
from cnn_rnn.cnn_model import CNN_Model
from data.hyper_parameters import HyperParams
from cnn_rnn.optimizer import Optimizer
import tensorflow as tf
import sys
import time

# load the data
dataset_path=sys.argv[1]
data = Dataset()
data.load_multivariate(dataset_prefix=dataset_path)

#start measuring time
start_time = time.time()

# get the hyper parameters for speech
hyperparams = HyperParams()
hyperparams.type = 'multivariate'
config = hyperparams.get_config(data, sys.argv)

# hide the test series measurements after the truncation point during testing, to make sure no method accesses it
data.mask_test_after(config['cnn_rnn:demanded_frac'])



# reset the graph for a new execution
tf.reset_default_graph()

# create final model with the best smoothing degree
model = CNN_Model(dataset=data, config=config)

# create an optimizer
opt = Optimizer(config=config)
opt.define_losses(dataset=data, model=model)
opt.optimize(dataset=data, model=model)

# stop measuring time
train_time = time.time() - start_time

print(data.dataset_name, train_time, sys.argv[2:], opt.best_early_results)

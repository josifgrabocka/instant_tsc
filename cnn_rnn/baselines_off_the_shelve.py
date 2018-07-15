from data.dataset import Dataset
import numpy as np
import sys
import time
import sklearn.svm

# load the data
dataset_path=sys.argv[1]


#start measuring time
start_time = time.time()

# create a model
model = sys.argv[2]

# the base model is a gaussian process

early_acc_results = []
for pct in np.arange(start=0.1, stop=1.11, step=0.1):

    # truncate train and test to use a percentage of the data
    data = Dataset()
    data.load_ucr_univariate_data(dataset_folder=dataset_path)
    truncated_data = data.create_truncated(pct)

    # leave only the series values and remove the derivatives, as GP does not support multivarate series
    truncated_data.X_train, truncated_data.X_test = truncated_data.X_train[:,:,0], truncated_data.X_test[:,:,0]
    # squeeze the dimensions
    truncated_data.X_train, truncated_data.X_test = np.squeeze(truncated_data.X_train), np.squeeze(truncated_data.X_test)

    if model == 'gp':
        gp_model = sklearn.gaussian_process.GaussianProcessClassifier(kernel=sklearn.gaussian_process.kernels.RBF())
        classifier = sklearn.multiclass.OneVsRestClassifier(estimator=gp_model)
    elif model == 'nn':
        classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)

    # fit the data
    classifier.fit(X=truncated_data.X_train, y=truncated_data.Y_train)

    # predict the test set
    Y_hat_test = classifier.predict(truncated_data.X_test)
    # measure the early accuracy
    early_acc_results.append(sklearn.metrics.accuracy_score(y_true=truncated_data.Y_test, y_pred=Y_hat_test))

# stop measuring time
train_time = time.time() - start_time

print(data.dataset_name, train_time, model, -1, early_acc_results)

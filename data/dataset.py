import numpy as np
import tensorflow as tf
import os
import random
from sklearn import preprocessing


class Dataset:

    def __init__(self):
        self.Y_train = None
        self.X_train = None
        self.Y_test = None
        self.X_test = None
        self.dataset_name = None
        self.num_classes = None
        self.series_length = None
        self.num_channels = None
        self.num_train_instances = None
        self.dataset_name = None

    def load_multivariate(self, dataset_prefix):

        X_train = np.load(dataset_prefix+"train_features.npy")
        Y_train = np.load(dataset_prefix+"train_labels.npy")

        X_test = np.load(dataset_prefix+"test_features.npy")
        Y_test = np.load(dataset_prefix+"test_labels.npy")

        Y_train = np.expand_dims(Y_train, axis=-1)
        Y_test = np.expand_dims(Y_test, axis=-1)

        self.num_train_instances = X_train.shape[0]
        self.series_length = X_train.shape[1]
        self.num_channels = X_train.shape[2]

        sorted_label_values = np.unique(Y_train)
        self.num_classes = sorted_label_values.size

        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder = onehot_encoder.fit(Y_train)

        Y_train_ohe = onehot_encoder.transform(Y_train)
        Y_test_ohe = onehot_encoder.transform(Y_test)

        self.X_train, self.Y_train = X_train, Y_train_ohe
        self.X_test, self.Y_test = X_test, Y_test_ohe

        self.dataset_name = os.path.basename(os.path.normpath(dataset_prefix))

        print(self.dataset_name, 'num_train_instances', self.num_train_instances, 'series_length', self.series_length,
              'num_channels', self.num_channels, 'num_classes', self.num_classes)


    def load_ucr_univariate_data(self, dataset_folder=None):

        # read the dataset name as the folder name
        self.dataset_name = os.path.basename(os.path.normpath(dataset_folder))

        # load the train and test data from files
        file_prefix = os.path.join(dataset_folder, self.dataset_name)
        train_data = np.loadtxt(file_prefix + "_TRAIN", delimiter=",")
        test_data = np.loadtxt(file_prefix + "_TEST", delimiter=",")

        # set train data
        self.Y_train = train_data[:, 0]
        self.X_train = train_data[:, 1:]
        self.num_train_instances = self.X_train.shape[0]

        # get the series length
        self.series_length = self.X_train.shape[1]

        # set the test data
        self.Y_test = test_data[:, 0]
        self.X_test = test_data[:, 1:]

        # get the label values in a sorted way
        sorted_label_values = np.unique(self.Y_train)
        self.num_classes = sorted_label_values.size

        print('Series length', self.series_length, ', Num classes', self.num_classes)

        # encode labels to a range between [0, num_classes)
        label_encoder = preprocessing.LabelEncoder()
        label_encoder = label_encoder.fit(self.Y_train)
        Y_train_encoded = label_encoder.transform(self.Y_train)
        Y_test_encoded = label_encoder.transform(self.Y_test)

        # convert the encoded labels to a 2D array of shape (num_instances, 1)
        Y_train_encoded = Y_train_encoded.reshape(len(Y_train_encoded), 1)
        Y_test_encoded = Y_test_encoded.reshape(len(Y_test_encoded), 1)

        # one-hot encode the labels
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder = onehot_encoder.fit(Y_train_encoded)
        self.Y_train = onehot_encoder.transform(Y_train_encoded)
        self.Y_test = onehot_encoder.transform(Y_test_encoded)

        # normalize the time series
        X_train_norm = preprocessing.normalize(self.X_train, axis=1)
        X_test_norm = preprocessing.normalize(self.X_test, axis=1)

        print('Train data', self.X_train.shape, 'Test data', self.X_test.shape)

        # add derivatives
        X_train_diff = np.diff(X_train_norm)
        X_test_diff = np.diff(X_test_norm)

        # replicate first column since diff has one less length than original
        X_train_diff_fc = np.expand_dims(X_train_diff[:, 0], 1)
        X_test_diff_fc = np.expand_dims(X_test_diff[:, 0], 1)

        X_train_diff = np.hstack([X_train_diff_fc, X_train_diff])
        X_test_diff = np.hstack([X_test_diff_fc, X_test_diff])

        X_train_diff = preprocessing.normalize(X_train_diff, axis=1)
        X_test_diff = preprocessing.normalize(X_test_diff, axis=1)

        self.X_train = np.stack([X_train_norm, X_train_diff], axis=-1)
        self.X_test = np.stack([X_test_norm, X_test_diff], axis=-1)

        self.num_channels = 2


    # draw a random set of instances from the training set
    def draw_batch(self, batch_size):
        # draw an arraw of random numbers from 0 to num rows in X_train
        random_row_indices = np.random.randint(0, self.num_train_instances, size=batch_size)

        X_batch = self.X_train[random_row_indices]
        Y_batch = self.Y_train[random_row_indices]

        # shrink/elongate randomly

        # slice the batch from the training set acc. to. the drawn row indices
        return X_batch, Y_batch




    # create a random tran and validation split from the training set
    def create_cv(self, pct):

        cv_data = Dataset()

        cv_data.dataset_name = self.dataset_name
        cv_data.num_classes = self.num_classes
        cv_data.series_length = self.series_length
        cv_data.num_channels = self.num_channels
        cv_data.num_train_instances = np.int(self.num_train_instances * pct)

        # create
        cv_data.X_train, cv_data.Y_train = self.draw_batch(cv_data.num_train_instances)
        cv_data.X_test, cv_data.Y_test = self.draw_batch(self.num_train_instances - cv_data.num_train_instances)

        return cv_data

    def create_truncated(self, pct):

        truncated_data = Dataset()

        truncated_data.dataset_name = self.dataset_name
        truncated_data.num_classes = self.num_classes
        truncated_data.num_channels = self.num_channels
        truncated_data.num_train_instances = self.num_train_instances
        truncated_data.series_length = np.int(np.ceil(self.series_length * pct))

        # truncate series
        truncated_data.X_train  = self.X_train[:,:truncated_data.series_length]
        truncated_data.X_test = self.X_test[:,:truncated_data.series_length]

        # destroy prev tensors
        self.X_train, self.X_test = None, None

        truncated_data.Y_train = self.Y_train
        truncated_data.Y_test = self.Y_test

        return truncated_data

    # set all test values after a threshold to zero
    def mask_test_after(self, demanded_fraction):

        non_masked_length = np.int(np.ceil(self.series_length * demanded_fraction))
        self.X_test[:, non_masked_length:,:].fill(0)


    def inflate_sliding_window(self, pct):

        inflated_data = Dataset()
        inflated_data.series_length = np.int(np.ceil(self.series_length * pct))

        inflated_data.dataset_name = self.dataset_name
        inflated_data.num_classes = self.num_classes
        inflated_data.num_channels = self.num_channels

        # set the test set directly
        inflated_data.X_test, inflated_data.Y_test = self.X_test, self.Y_test

        # number of subseries per series
        num_subseries = self.series_length - inflated_data.series_length + 1
        # number of training set
        inflated_data.num_train_instances = self.num_train_instances * num_subseries

        # create the train series and label matrices
        inflated_data.X_train = np.zeros(shape=(inflated_data.num_train_instances,
                                                inflated_data.num_channels,
                                                inflated_data.series_length))

        inflated_data.Y_train = np.zeros(shape=(inflated_data.num_train_instances))



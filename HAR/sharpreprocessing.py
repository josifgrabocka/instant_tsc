import numpy as np
import scipy.io as scio

acc_data_mat = scio.loadmat('/home/josif/ownCloud/research/tsc/baselines/shar_two/data/two_classes_data.mat')
acc_data = acc_data_mat['two_classes_data']

acc_labels_mat = scio.loadmat('/home/josif/ownCloud/research/tsc/baselines/shar_two/data/two_classes_labels.mat')
acc_labels = acc_labels_mat['two_classes_labels']

num_instances, total_length = acc_data.shape

num_channels = 3
num_multivariate_series = num_instances // 3

print(num_instances, num_multivariate_series, num_channels)

features = np.zeros(shape=(num_multivariate_series, total_length, num_channels))
labels = np.zeros(shape=(num_multivariate_series,))

for i in np.arange(num_multivariate_series):

    features[i] = np.swapaxes(acc_data[i*3:(i+1)*3], 0, 1)
    labels[i] = acc_labels[i*3][0]

    pass

# save the data to files
np.save('features.npy', features)
np.save('labels.npy', labels)

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensor import Tensor


class MNISTDataset:

    def load_mnist_data(self):

        # MNist is called once :)
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        data_train, data_test, labels_train, labels_test = train_test_split(mnist.data, mnist.target, test_size=10_000,
                                                                            random_state=0)
        # target = mnist.target.astype(int)

        labels_array_train = np.zeros((len(labels_train), 10), dtype=np.float32)
        for label_array_train, label_train in zip(labels_array_train, labels_train):
            label_array_train[int(label_train)] = 1
        labels_array_train = [Tensor(a) for a in labels_array_train]

        labels_array_test = np.zeros((len(labels_test), 10), dtype=np.float32)
        for label_array_test, label_test in zip(labels_array_test, labels_test):
            label_array_test[int(label_test)] = 1
        labels_array_test = [Tensor(a) for a in labels_array_test]

        return labels_array_train, labels_array_test, data_train, data_test

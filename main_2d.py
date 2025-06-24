from activations.relu_layer import ActivationsReLULayer
from layer.input_layer import InputLayer
from network import Network
from layer.fully_connected_layer import FullyConnectedLayer
from layer.flatten import Flatten
from tensor import Shape
from activations.softmax_layer import SoftmaxOutputLayer
from sgd_trainer import SGDTrainer
from activations.sigmoid_layer import ActivationsSigmoidLayer
from loss.cross_entropy import CrossEntropy
from mnist_dataset import MNISTDataset
from test_model import TestModel
from layer_2D.input_layer_2D import InputLayer2D
from layer_2D.convolution_layer import Conv2DLayer
from layer_2D.pooling_2D_layer import Pooling2DLayer, PoolingType

# Timer
import time


def train(data_train, labels_array_train):
    mnist = MNISTDataset()
    labels_array_train, labels_array_test, data_train, data_test = mnist.load_mnist_data()

    start_time = time.time()

    # Verhältnis out_shape abhängig von in_shape, kernel und stride:
    # out_shape = (in_shape - kernel) / stride + 1
    network = Network(
        input_layer=InputLayer2D(),
        layers=[
            Conv2DLayer(in_shape=Shape(axis=[28, 28, 1]), out_shape=Shape(axis=[26, 26, 4]),
                        kernel_size=Shape(axis=[3, 3]), num_filters=4),
            Pooling2DLayer(in_shape=Shape(axis=[26, 26, 4]), out_shape=Shape(axis=[13, 13, 4]),
                           kernel_size=Shape(axis=[2, 2]), pooling_type=PoolingType.MAX, stride=Shape(axis=[2, 2])),
            Conv2DLayer(in_shape=Shape(axis=[13, 13, 4]), out_shape=Shape(axis=[11, 11, 3]),
                        kernel_size=Shape(axis=[3, 3]), num_filters=3),
            Flatten(in_shape=Shape(axis=[11, 11, 3]), out_shape=Shape(axis=[363, ])),
            FullyConnectedLayer(in_shape=Shape(axis=[363, ]), out_shape=Shape(axis=[128, ])),
            ActivationsReLULayer(),
            FullyConnectedLayer(in_shape=Shape(axis=[128, ]), out_shape=Shape(axis=[32, ])),
            ActivationsReLULayer(),
            FullyConnectedLayer(in_shape=Shape(axis=[32, ]), out_shape=Shape(axis=[10, ])),
            SoftmaxOutputLayer()
        ],
        delta_params=[],
        caches=[],
    )
    #network.load_params(folder_path="networks_saved", name="Bestmodel2D")
    sgd = SGDTrainer(loss=CrossEntropy(), learningRate=0.003, batchSize=60_000, amountEpochs=10)
    sgd.optimize(network=network, data=data_train, labels=labels_array_train, singleBatches=True)
    return network


if __name__ == '__main__':
    mnist = MNISTDataset()
    labels_array_train, labels_array_test, data_train, data_test = mnist.load_mnist_data()

    train = input("Do you want to train a network (y) or load params (n)? ")

    if train == "y":
        start_time = time.time()
        network = train(data_train, labels_array_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        network.save_params(folder_path="networks_saved", name="Bestmodel2D")
    else:
        network = Network(
            input_layer=InputLayer2D(),
            layers=[],
            delta_params=[],
            caches=[],
        )
        network.load_params(folder_path="networks_saved", name="Bestmodel2D")

    test_model = TestModel()
    test_model.test_model(network=network, data_test=data_test, labels_array_test=labels_array_test)

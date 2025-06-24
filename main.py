from layer.input_layer import InputLayer
from network import Network
from layer.fully_connected_layer import FullyConnectedLayer
from tensor import Shape
from activations.softmax_layer import SoftmaxOutputLayer
from sgd_trainer import SGDTrainer
from activations.sigmoid_layer import ActivationsSigmoidLayer
from loss.cross_entropy import CrossEntropy
from mnist_dataset import MNISTDataset
from test_model import TestModel

# Timer
import time


def train(data_train, labels_array_train):
    network = Network(
        input_layer=InputLayer(),
        layers=[
            FullyConnectedLayer(in_shape=Shape(axis=[784, ]), out_shape=Shape(axis=[200, ])),
            ActivationsSigmoidLayer(),
            FullyConnectedLayer(in_shape=Shape(axis=[200, ]), out_shape=Shape(axis=[80, ])),
            ActivationsSigmoidLayer(),
            FullyConnectedLayer(in_shape=Shape(axis=[80, ]), out_shape=Shape(axis=[10, ])),
            SoftmaxOutputLayer()
        ],
        delta_params=[],
        caches=[],
    )
    # network.load_params(folder_path="networks_saved", name="Bestmodel")

    # train
    sgd = SGDTrainer(loss=CrossEntropy(), learningRate=0.05, batchSize=60_000,
                     amountEpochs=10)
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
        network.save_params(folder_path="networks_saved", name="Bestmodel")
    else:
        network = Network(
            input_layer=InputLayer(),
            layers=[],
            delta_params=[],
            caches=[],
        )
        network.load_params(folder_path="networks_saved", name="Bestmodel")


    test_model = TestModel()
    test_model.test_model(network=network, data_test=data_test, labels_array_test=labels_array_test)

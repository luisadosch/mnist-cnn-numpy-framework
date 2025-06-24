import time
from layer.fully_connected_layer import FullyConnectedLayer
from layer_2D.convolution_layer import Conv2DLayer
from loss.loss_interface import LossInterface
import numpy as np

from tensor import Tensor


class SGDTrainer:

    def __init__(self, loss: LossInterface, learningRate: float, amountEpochs: int, batchSize: int = 1,
                 shuffle: bool = True):
        self.loss = loss
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.amountEpochs = amountEpochs
        self.shuffle = shuffle

    def optimize(self, network, data: list[Tensor], labels: list[Tensor], singleBatches: bool):

        indices = np.random.choice(len(data), size=self.batchSize, replace=False)
        batch_data = [data[i] for i in indices]
        batch_labels = [labels[i] for i in indices]

        for epoch in range(self.amountEpochs):

            start_time = time.time()

            sum_loss = 0

            if singleBatches:
                for batch_data_single, batch_labels_single in zip(batch_data, batch_labels):
                    predicted = network.forward([batch_data_single])
                    # print("Expected Label: ", batch_labels[0].elements)
                    # print("Predicted Label: ", predicted[0].elements)
                    forward = self.loss.forward(predictions=predicted, labels=[batch_labels_single])
                    sum_loss += forward
                    # print(" Loss: ", forward)
                    self.loss.backward(predictions=predicted, labels=[batch_labels_single])

                    network.backprop()

                    for layer in network.layers:
                        if isinstance(layer, (FullyConnectedLayer, Conv2DLayer)):
                            layer.weights.elements -= self.learningRate * layer.weights.deltas
                            layer.bias.elements -= self.learningRate * layer.bias.deltas

                end_time = time.time()
                print(f"Epoch: {(str(epoch + 1))}   Loss: {sum_loss / len(data)}    Elapsed time: {(end_time - start_time):.2f} seconds")

            else:

                predicted = network.forward(batch_data)
                # print("Expected Label: ", batch_labels[0].elements)
                # print("Predicted Label: ", predicted[0].elements)
                forward = self.loss.forward(predictions=predicted, labels=batch_labels)
                print("Epoch: " + str(epoch+1) + " Loss: ", forward)
                self.loss.backward(predictions=predicted, labels=batch_labels)
                network.backprop()

                for layer in network.layers:
                    if isinstance(layer, (FullyConnectedLayer, Conv2DLayer)):
                        layer.weights.elements -= self.learningRate * layer.weights.deltas
                        layer.bias.elements -= self.learningRate * layer.bias.deltas

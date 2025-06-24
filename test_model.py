import numpy as np


class TestModel():

    def test_model(self, network, data_test, labels_array_test):

        correctly_predicted = 0
        total_number = 0

        for data, label in zip(data_test, labels_array_test):

            predicted = network.forward([data])
            predicted_label = np.argmax(predicted[0].elements)
            true_label = np.argmax(label.elements)

            if predicted_label == true_label:
                correctly_predicted += 1
            total_number += 1

        accuracy = correctly_predicted / total_number * 100
        print("------------------- Test Model ------------------------")
        print(f"Correctly Predicted: {correctly_predicted}")
        print(f"Total Number: {total_number}")
        print(f"Accuracy: {accuracy:.2f}%")

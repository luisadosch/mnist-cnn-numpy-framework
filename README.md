# DL Framework - Dosch - Esswein - Goebel

Dies ist ein einfaches, selbst entwickeltes Deep Learning Framework, das die Verarbeitung des MNIST-Datensatzes mit einem vollständig verbundenen neuronalen Netzwerk demonstriert. 

## Verwendung

### 1. Voraussetzungen
Stellen Sie sicher, dass alle benötigten Abhängigkeiten installiert sind. Diese sind in der Datei requirements.txt aufgeführt und beschränken sich im Wesentlichen auf numpy und scipy (für das Laden des Datensatzes).

Installation:

```shell
pip install -r requirements.txt
```

### 2. Training und Testen starten
Nach der Installation der Abhängigkeiten kann das Training und Testen des Netzwerks durch Ausführen des folgenden Skripts gestartet werden:

* Für das Fully Connected Netwerk der Aufgabe 1:

```shell
python main.py
```

* Für das Convolution Netzwerk der Aufgabe 2:

```shell
python main_2d.py
```

Dann können Sie sich entscheiden ob Sie das Netzwerk neu trainieren möchten oder ob Sie ein bereits gespeichertes Netzwerk ausführen möchten.

Dazu beantworten Sie die Frage im Terminal "Do you want to train a network (y) or load params (n)?" dementsprechend mit y or n.


Beim **Trainieren** des Models werden folgende Schritte automatisch durchgeführt:

* Laden des MNIST-Datensatzes

* Definition eines neuronalen Netzwerks mit mehreren Schichten

* Training des Netzwerks mit dem SGDTrainer

* Test des Modells auf den Testdaten

* Speichern der trainierten Modellparameter im Ordner networks_saved

Beim **Laden** des Modells werden: 

  * Es wird das gespeicherte Modell unter 

```python
network.load_params(folder_path="networks_saved", name="Bestmodel") beim Convolution network.load_params(folder_path="networks_saved", name="Bestmodel2D")
```

ausgeführt. 


## Ergebnisse Neural Network Fully Connected

### Best Model:
* **Architektur 1**

   | **Epoch** | **Loss** | **Elapsed Time (s)** |
   |-----------|----------|----------------------|
   | 1         | 0.2545   | 36.17                |
   | 2         | 0.1100   | 36.22                |
   | 3         | 0.0700   | 36.02                |
   | 4         | 0.0465   | 36.50                |
   | 5         | 0.0323   | 35.99                |
   | 6         | 0.0221   | 35.93                |
   | 7         | 0.0162   | 35.85                |
   | 8         | 0.0123   | 36.10                |
   | 9         | 0.0083   | 62.59                |
   | 10        | 0.0067   | 33.19                |


### Architektur 1:
 * **Layers:** Input (784) --> Fully Connected (200) --> Sigmoid --> Fully Connected (80) --> Sigmoid --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.05
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0067
   * Trainings-Zeit: 384.58 Sekunden
 *  **Test Genauigkeit:** 97.53%

### Architektur 2:
 * **Layers:** Input (784) --> Fully Connected (200) --> ReLu --> Fully Connected (80) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.001
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.067
   * Trainings-Zeit: 367,50 Sekunden
 *  **Test Genauigkeit:** 95,46%

### Architektur 3:
 * **Layers:** Input (784) --> Fully Connected (512) --> Sigmoid --> Fully Connected (256) --> Sigmoid --> Fully Connceted (128) --> Sigmoid --> Fully Connected (80) --> Sigmoid --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.001
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.136
   * Trainings-Zeit: 1401,63 Sekunden
 *  **Test Genauigkeit:** 95,07%

### Architektur 4:
 * **Layers:** Input (784) --> Fully Connected (512) --> Sigmoid --> Fully Connected (256) --> ReLU --> Fully Connceted (128) --> Sigmoid --> Fully Connected (80) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.001
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0673
   * Trainings-Zeit: 1362.74 Sekunden
 *  **Test Genauigkeit:** 94,68%

### Architektur 5:
 * **Layers:** Input (784) --> Fully Connected (80) --> Sigmoid -->  Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.001
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.225
   * Trainings-Zeit: 117.33 Sekunden
 *  **Test Genauigkeit:** 92.61%

### Architektur 6:
 * **Layers:** Input (784) --> Fully Connected (80) --> Sigmoid -->  Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.05
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.015
   * Trainings-Zeit: 115.00 Sekunden
 *  **Test Genauigkeit:** 96.81%

### Architektur 7:
 * **Layers:** Input (784) --> Fully Connected (80) --> ReLu -->  Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.001
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.132
   * Trainings-Zeit: 113.13 Sekunden
 *  **Test Genauigkeit:** 95.16%



## Ergebnisse CNN

### Best Model:
* **Architektur 7**

   | **Epoch** | **Loss** | **Elapsed Time (s)** |
   |-----------|----------|----------------------|
   | 1         | 0.1841   | 207.29               |
   | 2         | 0.0709   | 189.53               |
   | 3         | 0.0428   | 172.02               |
   | 4         | 0.0277   | 179.38               |
   | 5         | 0.0179   | 179.42               |
   | 6         | 0.0124   | 186.69               |
   | 7         | 0.0087   | 180.04               |
   | 8         | 0.0079   | 177.37               |
   | 9         | 0.0093   | 186.28               |
   | 10        | 0.0068   | 190.13               |


### Architektur 1 (Small Learning Rate)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,2) --> Pooling_Max (13,13,2) --> Flatten (338) --> Fully Connected (200) --> ReLu --> Fully Connected (80) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.0002
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0745
   * Trainings-Zeit: 569.53 Sekunden
 *  **Test Genauigkeit:** 96.49%

### Architektur 2 (Least Amount of Filters)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,2) --> Pooling_Max (13,13,2) --> Flatten (338) --> Fully Connected (200) --> ReLu --> Fully Connected (80) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.002
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0121
   * Trainings-Zeit: 614.24 Sekunden
 *  **Test Genauigkeit:** 97.20%

### Architektur 3 (Standard Model)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,3) --> Pooling_Max (13,13,3) --> Conv2D (11,11,3) --> Flatten (363) --> Fully Connected (128) --> ReLu --> Fully Connected (32) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.003
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0223
   * Trainings-Zeit: 777.90 Sekunden
 *  **Test Genauigkeit:** 97.84%

### Architektur 4 (Lower Learning Rate, More Filters, More Dense Nodes)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,4) --> Pooling_Max (13,13,4) --> Conv2D (11,11,4) --> Flatten (484) --> Fully Connected (128) --> ReLu --> Fully Connected (64) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.002
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0264
   * Trainings-Zeit: 760.81 Sekunden
 *  **Test Genauigkeit:** 97.26%

### Architektur 5 (High Learning Rate)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,4) --> Pooling_Max (13,13,4) --> Conv2D (11,11,4) --> Flatten (484) --> Fully Connected (128) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.006
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0374
   * Trainings-Zeit: 732.03 Sekunden
 *  **Test Genauigkeit:** 97.54%

### Architektur 6 (Multiple Conv Layer)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,4) --> Pooling_Max (13,13,4) --> Conv2D (11,11,4) --> Conv2D (9,9,2) --> Flatten (162) --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.002
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.1324
   * Trainings-Zeit: 720.31 Sekunden
 *  **Test Genauigkeit:** 94.79%

### Architektur 7 (Large Dense Layers)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,4) --> Pooling_Max (13,13,4) --> Flatten (676) --> Fully Connected (512) --> ReLu --> Fully Connected (256) --> ReLu --> Fully Connected (64) --> ReLu --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.002
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0068
   * Trainings-Zeit: 1850.96 Sekunden
 *  **Test Genauigkeit:** 98.15%

### Architektur 8 (Sigmoid)
 * **Layers:** Input (28,28,1) --> Conv2D (26,26,4) --> Pooling_Max (13,13,4) --> Conv2D (11,11,4) --> Flatten (484) --> Fully Connected (128) --> Sigmoid --> Fully Connected (64) --> Sigmoid --> Fully Connected (10 Klassen) --> Softmax
 * **Training:** 
   * Lern-Rate: 0.002
   * Loss: Cross Entropy
   * Loss (Epoche 10): 0.0549
   * Trainings-Zeit: 1010.21 Sekunden
 *  **Test Genauigkeit:** 97.07%


 ## Vergleich: Fully Connected vs Convolution

 * Die Trainingszeit des Convolutions ist länger als des FC
 * Die Accuracy des Convolution ist im Durchschnitt höher als des FC
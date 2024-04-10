import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score

train_file_name = "mnist_train.csv"
test_file_name = "mnist_test.csv"
training_path = os.path.dirname(__file__) + "\\" + train_file_name
testing_path = os.path.dirname(__file__) + "\\" + test_file_name
training_dataset = pd.read_csv(training_path)
testing_dataset = pd.read_csv(testing_path)

#Change training and testing dataset size
training_size = 1200
testing_size = 300

#hyperparameter values
k_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

class KNN:
    def __init__(self, k):
        self.k = k
        self.training_vectors = []
        self.training_labels = []
        self.distances = []

    def make_prediction(self, test_point):
        self.distances = []
        vectors = self.training_vectors
        labels = self.training_labels
        for data_point, label in zip(vectors, labels):
            self.distances.append((self.euclidean_dist(test_point, data_point), label))

        sorted_distances = sorted(self.distances)
        k_nearest_neighbors = np.array(sorted_distances[:self.k], dtype=object)
        label, frequency = np.unique(k_nearest_neighbors[:,1], return_counts=True)
        return label[frequency.argmax()]

    def euclidean_dist(self, vector_1, vector_2):
        return np.sqrt(np.sum((vector_1 - vector_2)**2))
   
    def addData(self, training_vectors, training_labels):
        self.training_vectors = training_vectors
        self.training_labels = training_labels 
    

#pixel values are normilised
def preprocess_data(vector):
    vector = vector.reshape(vector.shape[0], -1) / 255.0
    return vector

def execute_knn():
    # #collect training data
    training_vectors = preprocess_data(training_dataset.iloc[:training_size, 1:].values)
    training_labels = training_dataset.iloc[:training_size, 0].values   

    # #collect testing data
    testing_vectors = preprocess_data(testing_dataset.iloc[:training_size, 1:].values)
    testing_labels = testing_dataset.iloc[:training_size, 0].values

    knn = KNN(k=5)
    knn.addData(training_vectors, training_labels)

    #test the model
    predictions = []
    
    for data_point in testing_vectors:
        predicted_label = knn.make_prediction(data_point)
        predictions.append(predicted_label)
    predictions = np.array(predictions)

    #Calculate and display performance metrics
    accuracy = "{:.4f}".format(((predictions == testing_labels).sum()/testing_vectors.shape[0])*100)
    precision = "{:.4f}".format(precision_score(testing_labels, predictions, average='macro')*100)
    fOneScore = "{:.4f}".format(f1_score(testing_labels, predictions, average='macro')*100)
    print("Model accuracy:\t\t" +accuracy+"%")
    print("Model precision:\t" +precision+"%")
    print("Model f1-score:\t\t" +fOneScore+"%")

    print("################ Hyperparamter Optimisation results #####################")

    best_accuracy = 0
    best_f1 = 0
    best_precision = 0
    best_k = 0

    for k in k_values:
        knn = KNN(k=k)
        knn.addData(training_vectors, training_labels)

        #test the model
        predictions = []
    
        for data_point in testing_vectors:
            predicted_label = knn.make_prediction(data_point)
            predictions.append(predicted_label)
        predictions = np.array(predictions)

        #Calculate and display performance metrics
        accuracy = "{:.4f}".format(((predictions == testing_labels).sum()/testing_vectors.shape[0])*100)
        precision = "{:.4f}".format(precision_score(testing_labels, predictions, average='macro')*100)
        fOneScore = "{:.4f}".format(f1_score(testing_labels, predictions, average='macro')*100)
        
        if (float(accuracy) > float(best_accuracy)):
                best_k = k
                best_accuracy = accuracy
                best_f1 = fOneScore
                best_precision = precision

    print("Model accuracy:\t\t" +best_accuracy+"%")
    print("Model precision:\t" +best_precision+"%")
    print("Model f1-score:\t\t" +best_f1+"%\n")    
    print("\n\nHyperparameters used:  K=" + str(best_k)) 

    # Analyse predictions and actual values
    # for i in range(0,testing_size+1):
    #     print("Prediction: " + str(predictions[i]) + "\tActual: " + str(testing_labels[i]))
    

execute_knn()
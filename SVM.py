import os
import pandas as pd
from sklearn import svm 
from sklearn.metrics import precision_score, f1_score


train_file_name = "mnist_train.csv"
test_file_name = "mnist_test.csv"
training_path = os.path.dirname(__file__) + "\\" + train_file_name
testing_path = os.path.dirname(__file__) + "\\" + test_file_name
training_dataset = pd.read_csv(training_path)
testing_dataset = pd.read_csv(testing_path)

#set training and testing size
training_size = 1200
testing_size = 300

#Hyperparameter values
c_values = {1, 3, 10}
kernel_values = {'rbf', 'linear'}
gamma_values = {'scale', 'auto'}

def preprocess_data(vector):
    vector = vector.reshape(vector.shape[0], -1) / 255.0
    return vector

# #collect training data
training_vectors = preprocess_data(training_dataset.iloc[:training_size, 1:].values)
training_labels = training_dataset.iloc[:training_size, 0].values

# #collect testing data
testing_vectors = preprocess_data(testing_dataset.iloc[:training_size, 1:].values)
testing_labels = testing_dataset.iloc[:training_size, 0].values

#Create and train the model
svm_model = svm.SVC(kernel='rbf', C=1, gamma='auto')
svm_model.fit(training_vectors, training_labels)
predicted_labels = svm_model.predict(testing_vectors)

#Calculate and display performance metrics
accuracy = "{:.4f}".format(((predicted_labels == testing_labels).sum()/testing_vectors.shape[0])*100)
precision = "{:.4f}".format(precision_score(testing_labels, predicted_labels, average='macro')*100)
fOneScore = "{:.4f}".format(f1_score(testing_labels, predicted_labels, average='macro')*100)
print("Model accuracy:\t\t" +accuracy+"%")
print("Model precision:\t" +precision+"%")
print("Model f1-score:\t\t" +fOneScore+"%\n\n")

print("################ Hyperparamter Optimisation results #####################")

best_accuracy = 0
best_precision = 0
best_f1 = 0
best_c = 0
best_gamma = '' 
best_kernel = ''

for value in c_values:
    for gamma in gamma_values:
        for kernel in kernel_values:
            #Create and train the model
            svm_model = svm.SVC(kernel=str(kernel), C=value, gamma=str(gamma))
            svm_model.fit(training_vectors, training_labels)
            predicted_labels = svm_model.predict(testing_vectors)

            #Calculate and display performance metrics
            accuracy = "{:.4f}".format(((predicted_labels == testing_labels).sum()/testing_vectors.shape[0])*100)
            precision = "{:.4f}".format(precision_score(testing_labels, predicted_labels, average='macro')*100)
            fOneScore = "{:.4f}".format(f1_score(testing_labels, predicted_labels, average='macro')*100)

            if (float(accuracy) > float(best_accuracy)):
                best_c = value
                best_gamma = gamma
                best_kernel = kernel

                best_accuracy = accuracy
                best_f1 = fOneScore
                best_precision = precision

print("Model accuracy:\t\t" +best_accuracy+"%")
print("Model precision:\t" +best_precision+"%")
print("Model f1-score:\t\t" +best_f1+"%\n")    
print("\n\nHyperparameters used:  C=" + str(best_c) + ",     kernel=" + best_kernel + ",      gamma=" + best_gamma)           

#Analyse predictions and actual values
# for i in range(0,testing_size+1):
#     print("Prediction: " + str(predicted_labels[i]) + "\tActual: " + str(testing_labels[i]))

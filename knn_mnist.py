# K-Nearest Neighbor Classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime as dt
import seaborn as sns
import imageio
import os

# Plotting results method
def plot_results(classifier, test_labels, predicted_test):
    
    #plot classification report for test data
    tf.logging.info("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_labels, predicted_test))) 

    #print confusion matrix of model performance
    cm = metrics.confusion_matrix(test_labels, predicted_test)
    print(cm)

    #get accuracy measure for performance
    accuracy = metrics.accuracy_score(test_labels, predicted_test)
    
    #print classification accuracy for test data
    tf.logging.info("Accuracy={}".format(accuracy))
    
    #plot confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size = 15);
    plt.show()

# Read data method
def read_data():
    
    #read in training data set and labels
    folder = "../CMP9137M_14468387_Assessment_Item_1_Code/digits-train-5000/"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    train_data = []
    train_labels = []
    row = 0
    for _file in onlyfiles:
        train_data.append(imageio.imread(folder + _file))
        label_in_file = _file.find("_")
        train_labels.append(int(_file[0:label_in_file]))
        row = row + 1
        
    #append in validation data and labels to training set, due to cross-validation being used a larger training dataset is benefitial
    folder = "../CMP9137M_14468387_Assessment_Item_1_Code/digits-validation-1000/"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    for _file in onlyfiles:
        train_data.append(imageio.imread(folder + _file))
        label_in_file = _file.find("_")
        train_labels.append(int(_file[0:label_in_file]))

    train_data = np.array(train_data).reshape(6000, 784)
    
    #read in test set and labels
    folder = "../CMP9137M_14468387_Assessment_Item_1_Code/digits-test-500/"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    test_data = []
    test_labels = []
    for _file in onlyfiles:
        test_data.append(imageio.imread(folder + _file))
        label_in_file = _file.find("_")
        test_labels.append(int(_file[0:label_in_file]))

    test_data = np.array(test_data).reshape(500, 784)
    test_labels = np.array(test_labels) 
    
    #return the values to main for training
    return train_data, train_labels, test_data, test_labels

def main():
    
    #call read data function to read in dataset
    train_data, train_labels, test_data, test_labels = read_data()
    
    #initialise values for k and assign them to param_grid for gridsearchcv
    k = np.arange(20)+1
    param_grid = {'n_neighbors': k}

    #create classifiers from specified k values
    classifier = GridSearchCV(KNeighborsClassifier(), param_grid)

    #fit to training data
    best_model = classifier.fit(train_data, train_labels)

    #print the best k value
    print('Best k:', best_model.best_estimator_.get_params()['n_neighbors'])
    
    #retrain with best k value
    classifier = KNeighborsClassifier(n_neighbors=best_model.best_estimator_.get_params()['n_neighbors'])

    #start timer to calculate fitting time
    start_time = dt.datetime.now()
    tf.logging.info('Start learning at {}'.format(str(start_time)))
       
    #fit KNN regression classifier to training data
    classifier.fit(train_data, train_labels)
    
    #store end time
    end_time = dt.datetime.now() 
    tf.logging.info('Stop learning {}'.format(str(end_time)))

    #output fitting time 
    elapsed_time= end_time - start_time
    tf.logging.info('Elapsed learning {}'.format(str(elapsed_time)))
    
    #predict the value of the test data with KNN model
    predicted_test = classifier.predict(test_data)
    
    #plot results
    plot_results(classifier, test_labels, predicted_test)

if __name__ == "__main__":
    main()
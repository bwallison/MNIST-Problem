# CNN Classification

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
import seaborn as sns
import os
import imageio
import numpy as np

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

def plot_training(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def read_data():
    
    #define num of classes and images
    num_classes = 10
    img_rows, img_cols = 28, 28
    
    #read in training data set and labels
    folder = "../CMP9137M_14468387_Assessment_Item_1_Code/digits-train-5000/"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    train_data = []
    train_labels = []
    for _file in onlyfiles:
        train_data.append(imageio.imread(folder + _file))
        label_in_file = _file.find("_")
        train_labels.append(int(_file[0:label_in_file]))

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    #read in validation set and labels
    folder = "../CMP9137M_14468387_Assessment_Item_1_Code/digits-validation-1000/"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    val_data = []
    val_labels = []
    for _file in onlyfiles:
        val_data.append(imageio.imread(folder + _file))
        label_in_file = _file.find("_")
        val_labels.append(int(_file[0:label_in_file]))

    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
       
    #read in test set and labels
    folder = "../CMP9137M_14468387_Assessment_Item_1_Code/digits-test-500/"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    test_data = []
    test_labels = []
    for _file in onlyfiles:
        test_data.append(imageio.imread(folder + _file))
        label_in_file = _file.find("_")
        test_labels.append(int(_file[0:label_in_file]))

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    #restructure data for cnn
    if K.image_data_format() == 'channels_first':
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
        val_data = val_data.reshape(val_data.shape[0], 1, img_rows, img_cols)
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else :
        train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
        val_data = val_data.reshape(val_data.shape[0], img_rows, img_cols, 1)
        test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
      
    train_data = train_data.astype('float32')
    val_data = val_data.astype('float32')
    test_data = test_data.astype('float32')

    train_data /= 255
    val_data /= 255
    test_data /= 255    
    
    #convert labels to catergorical
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    val_labels = keras.utils.to_categorical(val_labels, num_classes)    

    print('train_data shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    print(val_data.shape[0], 'test samples')
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels, input_shape, num_classes

def main():
    
    #define batch size and number of training epochs
    batch_size = 128
    epochs = 12
    
    #call read data function to read in dataset
    train_data, train_labels, val_data, val_labels, test_data, test_labels, input_shape, num_classes = read_data()
  
    #define cnn model
    model = Sequential()
    
    #add first and second convolutional layers
    model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    
    #add pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #flatten
    model.add(Flatten())
    
    #add two dense layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    #compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    #fit training data to cnn using the validation data set as a test
    history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(val_data, val_labels))
    score = model.evaluate(val_data, val_labels, verbose =0)
    
    #print final score accuracy
    print('Validation Test loss:', score[0])
    print('Validation Test accuracy:', score[1])
    print('--- %s seconds ---" % (time.time() - start_time)')

    #plot training
    plot_training(history)

    #predict the test set data
    predicted_test = model.predict_classes(test_data, batch_size=batch_size, verbose=0).tolist()
    
    #plot results
    plot_results(model, test_labels, predicted_test)

if __name__ == "__main__":
    main()
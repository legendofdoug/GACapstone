# Capstone Technical Report: Hand Gesture Recognition


## Problem Statement
Given a large Dataset of Hand Gesture Images gathered, we want to able recognize the hand sign that is being shown on the image.

## Project Folder Layout

csv: Where the splitted datasets will go in CSV form.
data: Location of the dataset from Kaggle.
models: Location of where the Notebooks will save models in case of a failure.
notebooks: Location of the Notebooks.
There are only two notebooks.
01: EDA + Importing Data
- This is where the Data is imported as images and then exported as integer arrays. It is then split into 4 different CSVs so that RAM is not over filled.
02: Modeling
- Where the CSVs are imported one by one to train models. It is here the Neural Networks are trained. Results analysis is also done here.

## Exploratory Data Analysis:
Hand Gesture Recognition is a Kaggle Data taken from the following link:

https://www.kaggle.com/gti-upm/leapgestrecog

Please note that I did not include the dataset in this project folder. Download the dataset from kaggle and extract it into the data folder. Rename the extracted folder "hands_train"


The following dataset contains 10 test subjects doing 10 hand signs. Each hand sign has 200 images each. This comes out to 20,000 images, each one at 640 by 240 pixels. This generates over 3 Billion data points. This causes huge amounts of problems as we try to run it through a model, as it would cause ram issues regularly. Each machine has varying amounts of ram, as such the data set should be broken down differently per machine.

Originally when data from images was being converted to integer values in matrix form, the information was saved into a Pandas dataframe. This was a huge mistake initially. Each row consisted of 158,400 columns per image. This was an enormous amount of data to apply to a data frame. Pandas dataframes also expand data usage on RAM by 6-7 fold. Even on workstation machines, RAM would eventually run out and other avenues of training models would have to be used. 
    
In the interest of simply familiarizing ourselves with Convolutional Neural Networks and Feed Forward Neural Networks., we need to simplify our data set. We removed half of the hand signs with only signs palm, L, fist, thumb and index remain.

When transcribing the images into matrix values, it was decided to remove ¾ of the pixels. This was done by removing every other column and row of pixels. The result is a 320 by 120 pixel image. A gaussian blur with a sigma of 5 was applied to the image to reconstruct it. The result is a data set that is only ¼ the size of the original data set. 

After initially splitting the data into 2 csvs, each at 500 MB, it was later decided to split them into 4 csvs to get faster loading times. Each file would roughly be 250 MB. 


## Modeling:

A primary reason why this particular project was the exploratory use of GPU usage on a neural network application. An Nvidia Geforce GTX 960m was used to do deep learning on the data set. Getting the GPU to function correctly with the Keras package took some time. 
The initial model used was the convolutional neural network. Various hyperparameters were tuned to try to find the best outcome. The best ended up being:

```PYTHON
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, 
                           verbose=1, mode='auto')
cnn_model=Sequential()
cnn_model.add(Conv2D(filters = 10,            # number of filters
                     kernel_size = 5,        # height/width of filter
                     activation='relu',      # activation function 
                     #kernel_regularizer=regularizers.l2(0.01),
                     input_shape=(120, 320,1))) # shape of input (image)
cnn_model.add(MaxPooling2D(pool_size=(2,2))) # dimensions of region of pooling
cnn_model.add(Conv2D(128,
                     kernel_size=3,
                     #kernel_regularizer=regularizers.l2(0.01),
                     activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
# Add a densely-connected layer with 128 neurons.
cnn_model.add(Dense(128, activation='relu',))
cnn_model.add(Dropout(.5))
# Add a densely-connected layer with 6 neurons.
cnn_model.add(Dense(y_train.shape[1], activation='relu'))
cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
# Fit model on training data
cnn_history = cnn_model.fit(X_train,
                        y_train,
                        batch_size=64,
                        validation_data=(X_test, y_test),
                        epochs=100,
                        verbose=1,
                        callbacks=[early_stop]
                        )
```

After some time tuning the model, a basic feed forward neural network (FFNN) was implemented. The code for which is below:

```PYTHON
model = Sequential()
model.add(Dense(128, # How many neurons do you want in your first layer.
                input_shape=(38400,),
                activation='relu'))
model.add(Dropout(.5))
model.add(Dense(y_train.shape[1], activation = 'softmax'))

model.compile (optimizer = 'adam', metrics = ['accuracy'], loss = 'categorical_crossentropy')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

history = model.fit(X_train_sc, y_train, 
                    validation_data=(X_test_sc, y_test), 
                    epochs=100, 
                    batch_size=512,
                    verbose = 1,
                   callbacks=[early_stop])
```



Interestingly enough, it was outperforming the Covolutional Neural Network (CNN) by quite a large margin. The FFNN classified up to 99.02% accuracy whereas the CNN only achieved a ~90% percent on its best run. The CNN’s testing/validation score was much higher than those of the training score which was very strange. This is normally not the case. There was likely an issue with the code that will have to be sorted out at a later date. 
Several other avenues of hyperparameter tuning was explored. This included grid search, randomized search and ANNs. The time invested in these did not pan out this time around and absorbed a great deal of time. 


## Results/Recommendations:

As of now, the model is not ready for any kind of use although it has a great start and has very been useful in a learning environment. It was clear that a mid-range laptop GPU is able to keep pace with a powerful laptop GPU (Nvidia GTX 960m VS an Intel 6700HQ) at a much lower power consumption and system heat. This project will likely be revisited after more research has been done with Neural Networks and more ways on how to properly tune its hyperparameters. 


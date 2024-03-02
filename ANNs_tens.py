import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, PReLU, ELU
from tensorflow.keras.layers import Dropout




dataset=pd.read_csv("C:/Users/ADMIN/Desktop/data_pipeline/Churn_Modelling.csv")
print(dataset.head())

# Divide the dataset into independent and dependent features

X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]

#print(X.head())
#print(y.head())

## Feature Engineering

geography=pd.get_dummies(X['Geography'])
gender=pd.get_dummies(X['Gender'])

#concatenate these variables with dataframe

X=X.drop(['Geography','Gender'],axis=1)

X=pd.concat([X,geography,gender],axis=1)


#train the ANN now

#splitting the dataset into Training set and Test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#print(X_train.shape)

#print(X_test.shape)

## Part 2 now lets create the ANN


# Lets initialize the ANN

classifier=Sequential()

# Adding the input layer

classifier.add(Dense(units=11,activation='relu'))

#adding the first hidden layer (Dense means layer)
classifier.add(Dense(units=7,activation='relu'))

classifier.add(Dropout(.2))

#adding the second hidden layer

classifier.add(Dense(units=6,activation='relu'))

classifier.add(Dropout(.3))

## Adding the output layer

classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

opt=tf.keras.optimizers.Adam(learning_rate=.01)

#model_history=classifier.fit(X_train,y_train,validation_split=.33,batch_size=10,epochs=1000)


#Early stopping 

early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model_history=classifier.fit(X_train,y_train,validation_split=.33,batch_size=10,epochs=1000,callbacks=early_stopping)

model_history.history.keys()


#Part 3-Making the predictions and evaluting the model

#Predicting the Test set results

y_pred=classifier.predict(X_test)

y_pred=(y_pred>=0.5)

# making the confusion matrix

cm=confusion_matrix(y_test,y_pred)
print(cm)

#calculate the accuracy

score=accuracy_score(y_pred,y_test)
print(score)

#get the weights

wgt=classifier.get_weights()

print(wgt)




print()
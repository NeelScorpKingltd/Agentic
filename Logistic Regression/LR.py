#A simple logostic Regression model to classify the MNIST dataset of handwritten digits. The code includes data loading, preprocessing, model training, and evaluation using accuracy and confusion matrix.

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score
import numpy as np  
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

#MNIST is a Public image dataset of handwritten digits, commonly used for training and testing in the field of machine learning. It contains 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

mnist=fetch_openml('mnist_784')

#mnist is a dictionaky with several keys, relevant for us are target and data
#mnist.data contains the pixel data for each image and mnist.target contains expected digit in that image 

print(mnist.keys())
#dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

print(mnist.data.shape) #(70000, 784) 70000 images and each image has 784 pixels (28*28)
print(mnist.target.shape) #(70000,) 70000 labels for each image

print(mnist.data.iloc[0].min(), mnist.data.iloc[0].max()) #0.0 255.0 pixel values are between 0 and 255

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(np.array(mnist.data[0:4]),np.array(mnist.target[0:4]))):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title('Label: ' + label, fontsize = 20)




#to shorten the training time we will use only 20000 samples
df_data=mnist.data.loc[:19999].copy()
df_labels=mnist.target.loc[:19999].copy()

#before training the model we are first required to normalize the data
#we will divide each pixel value by 255 to scale the values between 0 and 1

for col in df_data.columns:
    df_data[col] = df_data[col] / 255.0

"""#split the data into training and testing sets(Train 1)
train_image, test_image, train_label, test_label = train_test_split(df_data, df_labels,test_size=0.3, random_state=0)

train_image.reset_index(drop=True,inplace=True) 
test_image.reset_index(drop=True,inplace=True) 

train_label.reset_index(drop=True,inplace=True)
test_label.reset_index(drop=True,inplace=True)


logreg=LogisticRegression(solver='lbfgs')
logreg.fit(train_image,train_label)"""

#training the logistic regression model (Train 2)# Training loop with different random states (10 times)


for iteration in range(10):
    print(f"\n--- Training Iteration {iteration + 1} ---")
    
    # Split with different random_state each time
    train_image, test_image, train_label, test_label = train_test_split(
        df_data, df_labels, test_size=0.3, random_state=iteration
    )
    
    # Reset indices
    train_image.reset_index(drop=True, inplace=True) 
    test_image.reset_index(drop=True, inplace=True) 
    train_label.reset_index(drop=True, inplace=True)
    test_label.reset_index(drop=True, inplace=True)
    
    # Train model
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    logreg.fit(train_image, train_label)
    
    


train_image, test_image, train_label, test_label = train_test_split(df_data, df_labels,test_size=0.3, random_state=8)

"""train_image.reset_index(drop=True,inplace=True) 
test_image.reset_index(drop=True,inplace=True) 

train_label.reset_index(drop=True,inplace=True)
test_label.reset_index(drop=True,inplace=True)

logreg=LogisticRegression(solver='lbfgs')
logreg.fit(train_image,train_label)

#Testing accuracy
test_pred=logreg.predict(test_image)
accuracy=accuracy_score(test_label, test_pred)
print ("Testing Accuracy :", accuracy)"""

test_pred = logreg.predict(test_image)
accuracy = accuracy_score(test_label, test_pred)
print("Testing Accuracy:", accuracy)
   

#generate the confusion matrix to validate the accuracy
cm=confusion_matrix(test_label, test_pred)

disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)    
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Logistic Regression")
plt.show( )
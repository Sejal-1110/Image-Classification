#preprocessing
#1. Resizing 
#2.Flattening

import os
# play aorund the folder
import matplotlib.pyplot as plt
#Displaying the images
import numpy as np
#Numerical computing
from skimage.io import imread
#Reading the images
from skimage.transform import resize
#Not a specific sized

target = []
images = []
flat_data = []

DATADIR = 'D:\Projects\Machine Learning\images'
CATEGORIES = ['Anne Princess Royal','Camilla Duchess of Cornwalls','Charles Prince of Wales','Diana Princess of Wales','Kate middletons','Meghan Markle','Prince Harry','Prince Philip Duke','Prince William','Queen Elizabeth']

for category in CATEGORIES:
    class_num = CATEGORIES.index(category) #Label Encoding the values
    path = os.path.join(DATADIR,category) #Create path to use all the images
    for img in os.listdir(path):
      img_array = plt.imread(os.path.join(path,img))
      #print(img_array.shape)
      #plt.imshow(img_array)

      img_resized = resize(img_array,(150,150,3))  #Normalizes the Value from0 to 1
      flat_data.append(img_resized.flatten())
      images.append(img_resized)
      target.append(class_num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)




len(flat_data[0])



target




unique,count = np.unique(target,return_counts = True)
plt.bar(CATEGORIES,count)




#Splitt Data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(flat_data,target,test_size = 0.3,random_state = 109)





from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid = [
    {'C' : [1,10,100,1000], 'kernel' :['linear']},
    {'C' : [1,10,100,1000],'gamma' : [0.001,0.0001] ,'kernel' :['rbf']},
]
svc = svm.SVC(probability=True)

clf = GridSearchCV(svc,param_grid)
clf.fit(x_train,y_train)
              


y_pred = clf.predict(x_test)
y_pred




y_test


from sklearn.metrics import accuracy_score, confusion_matrix



accuracy_score(y_pred,y_test)



confusion_matrix(y_pred,y_test)


#Save the model using pickle library
import pickle
pickle.dump(clf,open('img_model.p','wb'))



model = pickle.load(open('img_model.p',rb))



#Testing a brand new Image



flat_data = []
url = input('Enter your URL')
img = imread(url)
img_resized = resize(img,(250,250,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]]
print(f'PREDICTED OUTPUT: {y_out}')








      
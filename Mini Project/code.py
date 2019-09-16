#Importing libraries
import pandas as pd

#reading data and giving column names
data = pd.read_csv("adult_data.csv", names = ["Age", "Workclass", "Fnlwgt","Educaion","EducationNo","MaritalStatus","Occupation","Relationship","Race","Sex","CapitalGain","CapitalLoss","Hoursperweek","Country","Earnings"])

#To study the data
print(data.describe())          #To get details of data table
print(data.info())              #To check for null values
print('The Data shape before preprocessing is :', data.shape)      #To see the no.of rows and columns
dtype = data.dtypes             #Assigning datatypes to variable

#Converting categorical values to numerical values
from sklearn.preprocessing import LabelEncoder
for i in range(0, 15):
    if dtype.iloc[i] == object:        #Checking if the datatype is object
        labelencoder = LabelEncoder()  #Assigning encoder function to variable
        data.iloc[:, i] = labelencoder.fit_transform(data.iloc[:, i].values)   #Converts workclass to numerals
        
#Outlier Dectection
import seaborn as sns
sns.boxplot(x = data.columns['EducationNo'])

#Addressing missing values                       
import numpy as np            #Imporing necessary library
data.Occupation.replace(0, np.nan, inplace = True)  #Coverting 0 in Occupation to nan
data.Workclass.replace(0, np.nan, inplace = True)   #Coverting 0 in Occupation to nan 
data = data.fillna(data.mean())                     #Replacing all nan with mean
print('The Data shape after preprocessing is :', data.shape, '\n')

#Visualising the data
import matplotlib.pyplot as plt    #Importing necessary libraries
freqgraph = data.select_dtypes(include=['int'])     #Selecting variables to plot
freqgraph.hist(figsize=(10,10))     #Plotting the histogram
plt.show()

#Data Correlation
import matplotlib.pyplot as plt                            #Importing necessary libraries
import seaborn as sns
corrmatrix = data.corr()                                   #Assigning correlation function to variable
plt.subplots(figsize =(12, 9)) 
sns.heatmap(corrmatrix, vmin = 0, vmax = .8, annot = True, cmap = "YlGnBu", linewidths = 0.1)        #Plotting correlation heatmap

#Splitting data into training and testing set
from sklearn.model_selection import train_test_split
X = data[['EducationNo','Age','Hoursperweek','Sex','CapitalGain','CapitalLoss']].values    #Defining independent variables 
y = data[['Earnings']].values              #Defining dependent variable                       
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.3, random_state = 21, stratify = y)      #Dividing the dataset into test set and training set

# Feature Scaling
from sklearn.preprocessing import StandardScaler   #Importing necessary libraries
sc = StandardScaler()                              #Assigning scalaer function to variable
X_train = sc.fit_transform(X_train)                #Fitting the transform to training model
X_test = sc.transform(X_test)                      #Transforming the test set

#Writting preprocessed data to a new file
dF1 = pd.DataFrame(X_train)    #Loading train dataset to a variable
dF2 = pd.DataFrame(X_test)     #Loading test dataset to a variable
dF = pd.concat([dF1, dF2])     #Merging train and test sets
dF.to_csv('preprocessed adult data.csv', index = False)  #Writing to new file
dF = pd.read_csv('preprocessed adult data.csv', names = ['EducationNo','Age','Hoursperweek','Sex','CapitalGain','CapitalLoss'])   #Reading new file and giving column name
dF= dF.drop([0])    #Removing unwanted first row
dF.to_csv('preprocessed adult data.csv', index = False)   #Rewriting data 

#Decision Tree model
from sklearn import tree                          #Importing libraries
from sklearn import metrics
clf = tree.DecisionTreeClassifier()               #Assigning decision tree model to variable
clf.fit(X_train, y_train.ravel())                 #Fitting the model to training datasets
predn = clf.predict(X_test)                       #Using the model to predict the result
from sklearn.metrics import confusion_matrix      #Importing libraries for confusion matrix
cmDecsnTree = confusion_matrix(y_test, predn)     #Assigning confusion matrix to variable
print('The accuracy of Decision tree model is',metrics.accuracy_score(predn, y_test) * 100, '%')

#Logistic Regression model
from sklearn.linear_model import LogisticRegression    #Importing necessary libraries
classifier = LogisticRegression(random_state = 0, solver='lbfgs')  #Assigning logistic regression model to variable  
classifier.fit(X_train, y_train.ravel())      #Fitting the model to training datasets
model_pred = classifier.predict(X_test)       #Using the model to predict the result
cmLogstcRgrsn = confusion_matrix(y_test, model_pred)    #Assigning confusion matrix to variable
print('The accuracy of Logistic regression model is',metrics.accuracy_score(model_pred, y_test) * 100, '%')

#kNN model
from sklearn.neighbors import KNeighborsClassifier       #Importing libraries
model = KNeighborsClassifier(n_neighbors = 18, metric = 'manhattan', p = 2)     #Assingning kNN model to variable
model.fit(X_train, y_train.ravel())           #Fitting the model to training datasets
prediction = model.predict(X_test)            #Using the model to predict the result
cmKnn = confusion_matrix(y_test, prediction)  #Assigning confusion matrix to variable
print('The accuracy of the kNN is', metrics.accuracy_score(prediction, y_test) * 100, '%')

#SVM
from sklearn.svm import SVC      #Importing necessary libraries
classifier = SVC(kernel = 'linear', random_state = 0)    #Assigning SVM model to variable
classifier.fit(X_train, y_train.ravel())    #Fitting the model to training datasets 
y_pred = classifier.predict(X_test)         #Using the model to predict the result
cmSvm = confusion_matrix(y_test, y_pred)    #Assigning confusion matrix to variable     
print('The accuracy of the SVM is', metrics.accuracy_score(y_pred, y_test) * 100, '%')

#Kernel SVM
from sklearn.svm import SVC    #Importing necessary libraries
svclassifier = SVC(kernel='rbf', degree=8)      #Assigning Kernel SVM model to variable
svclassifier.fit(X_train, y_train.ravel())      #Fitting the model to training datasets
y_pre = svclassifier.predict(X_test)            #Using the model to predict the result
cmKernselSvm = confusion_matrix(y_test, y_pre)  #Assigning confusion matrix to variable
print('The accuracy of the Kernel SVM is', metrics.accuracy_score(y_pre, y_test) * 100, '%')

#Random forest model
from sklearn.ensemble import RandomForestClassifier  #Importing necessory libraries
regressor = RandomForestClassifier(n_estimators=100, random_state = 10)  #Assigning Random Forest model to variable
regressor.fit(X_train, y_train.ravel())    #Fitting the model to training datasets
y_predctn = regressor.predict(X_test)      #Using the model to predict the result
print('The accuracy of the Random Forest Model is', metrics.accuracy_score(y_predctn, y_test) * 100, '%\n')

#To display the maximum accuracy
maxacc = [metrics.accuracy_score(predn, y_test), metrics.accuracy_score(model_pred, y_test), metrics.accuracy_score(prediction, y_test), metrics.accuracy_score(y_pred, y_test), metrics.accuracy_score(y_pre, y_test), metrics.accuracy_score(y_predctn, y_test)]
print('The maximum accuracy is:', max(maxacc) * 100, '%')
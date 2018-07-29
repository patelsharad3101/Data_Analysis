# Titanic Dataset

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
dataset = [train_df, test_df]
print(train_df.columns.values)

#Removing Unwanted columns in dataset
print("Before", train_df.shape, test_df.shape, dataset[0].shape, dataset[1].shape)
train_df = train_df.drop(['PassengerId','Ticket', 'Cabin','Name','SibSp','Parch'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin','Name','SibSp','Parch'], axis=1)
dataset = [train_df, test_df]
"After", train_df.shape, test_df.shape, dataset[0].shape, dataset[1].shape

# Filling NAN values of Age and Fare by median values in Training and Test Dataset

median_value_train = train_df['Age'].median()
train_df['Age']=train_df['Age'].fillna(median_value_train)
median_value_test = test_df['Age'].median()
test_df['Age']=test_df['Age'].fillna(median_value_test)
median_value_test_fare = test_df['Fare'].median()
test_df['Fare']=test_df['Fare'].fillna(median_value_test_fare)

#Finding NAN from Embarked and filling with most freqhent value

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in dataset:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
# splitting Training and testing dataset into desired format    
X_train = train_df.drop("Survived", axis=1).copy()
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape    

# converting dataframe into values
X_train_val = X_train.iloc[:,:].values
X_test_val = X_test.iloc[:,:].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#training Set
labelencoder_X_1 = LabelEncoder()
X_train_val[:, 4] = labelencoder_X_1.fit_transform(X_train_val[:, 4])
labelencoder_X_2 = LabelEncoder()
X_train_val[:, 1] = labelencoder_X_2.fit_transform(X_train_val[:, 1])
onehotencoder_1 = OneHotEncoder(categorical_features = [4])
X_train_val = onehotencoder_1.fit_transform(X_train_val).toarray()
X_train_val = X_train_val[:, 1:]
#test Set
labelencoder_X_3 = LabelEncoder()
X_test_val[:, 4] = labelencoder_X_3.fit_transform(X_test_val[:, 4])
labelencoder_X_4 = LabelEncoder()
X_test_val[:, 1] = labelencoder_X_4.fit_transform(X_test_val[:, 1])
onehotencoder_2 = OneHotEncoder(categorical_features = [4])
X_test_val = onehotencoder_2.fit_transform(X_test_val).toarray()
X_test_val = X_test_val[:, 1:]


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 0)
logreg.fit(X_train_val, Y_train)
y_pred_logreg = logreg.predict(X_test_val)
acc_logreg = round(logreg.score(X_train_val, Y_train) * 100, 2)
acc_logreg

# Fitting KNN to the Training set

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train_val, Y_train)
y_pred_knn = knn.predict(X_test_val)
acc_knn = round(knn.score(X_train_val, Y_train) * 100, 2)
acc_knn

#Fitting SVM to the training Set
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 0)
svm.fit(X_train_val, Y_train)
y_pred_svm = svm.predict(X_test_val)
acc_svm = round(svm.score(X_train_val, Y_train) * 100, 2)
acc_svm

#Fitting Kernel SVM
from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', random_state = 0)
ksvm.fit(X_train_val, Y_train)
y_pred_ksvm = ksvm.predict(X_test_val)
acc_ksvm = round(ksvm.score(X_train_val, Y_train) * 100, 2)
acc_ksvm

#Fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X_train_val, Y_train)
y_pred_naive = naive.predict(X_test_val)
acc_naive = round(naive.score(X_train_val, Y_train) * 100, 2)
acc_naive


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decision.fit(X_train_val, Y_train)
y_pred_decision = decision.predict(X_test_val)
acc_decision = round(decision.score(X_train_val, Y_train) * 100, 2)
acc_decision

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)
rfr.fit(X_train_val, Y_train)
y_pred_rfr = rfr.predict(X_test_val)
acc_rfr = round(rfr.score(X_train_val, Y_train) * 100, 2)
acc_rfr

#Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(X_train_val, Y_train)
y_pred_linear_svc = linear_svc.predict(X_test_val)
acc_linear_svc = round(linear_svc.score(X_train_val, Y_train) * 100, 2)
acc_linear_svc

# Fitting Perceptron to the dataset
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train_val, Y_train)
y_pred_perceptron = perceptron.predict(X_test_val)
acc_perceptron = round(perceptron.score(X_train_val, Y_train) * 100, 2)
acc_perceptron

# Fitting Stochastic Gradient Descent Classifier to the dataset
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train_val, Y_train)
y_pred_sgd = sgd.predict(X_test_val)
acc_sgd = round(sgd.score(X_train_val, Y_train) * 100, 2)
acc_sgd


#Fitting ANN to the Dataset
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 6))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train_val, Y_train, batch_size = 10, nb_epoch = 100)


#Comparison of models

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Kernel SVM'],
    'Score': [acc_svm, acc_knn, acc_logreg, 
              acc_rfr, acc_naive, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision, acc_ksvm]})
models.sort_values(by='Score', ascending=False)

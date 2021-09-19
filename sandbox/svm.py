import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


WORK_DIR = '/home/alex/projects/petproject1/sandbox/data/'
train_data = pd.read_csv(WORK_DIR + "train.csv") #reading the csv files using pandas
test_data = pd.read_csv(WORK_DIR + "test.csv")



## Separating the X and Y variable

y = train_data['label']

## Dropping the variable 'label' from X variable 
X = train_data.drop(columns = 'label')

## Printing the size of data 
print(train_data.shape)



## Normalization

X = X/255.0
test_data = test_data/255.0

print("X:", X.iloc[0].shape)
print("test_data:", test_data.shape)



# scaling the features
#from sklearn.preprocessing import scale
#X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)


'''
# creating a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 10)

# specify range of hyperparameters
# Set the parameters by cross-validation
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [5,10]}]




# specify model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)
# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))

Fitting 5 folds for each of 6 candidates, totalling 30 fits
The best test score is 0.9448809523809523 corresponding to hyperparameters {'C': 10, 'gamma': 0.001}
'''

# model with optimal hyperparameters

# model
model = SVC(C=10, gamma=0.001, kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metrics
print("accuracy", accuracy_score(y_test, y_pred), "\n")
print(confusion_matrix(y_test, y_pred), "\n")

from joblib import dump
dump(model, WORK_DIR + 'svm.joblib') 

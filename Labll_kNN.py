
# %% [markdown]
# Question: Is there a correlation between estimated median SAT score and student completion/graduation rate?

# %% 
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
cc_df = pd.read_csv("cc_institution_details.csv")


# %% [markdown]
# Step 1/2: Define target variable and kNN model
# %%
#kNN requries a categorical target. Create binary target variable by converting graduation rate into High Completion (1) vs Low Completion (0).
cc_df['high_completion'] = [
    1 if cc_df['grad_100_value'][i] > cc_df['grad_100_value'].median() 
    else 0 
    for i in range(len(cc_df))
]

#define X and y by first filling in missing values with median 
columns = ['med_sat_value', 'high_completion']
for col in columns:
    cc_df[col].fillna(cc_df[col].median(), inplace=True)

cc_df.fillna({'med_sat_value': cc_df['med_sat_value'].median()}, inplace=True) #fill missing values in med_sat_value with median. for loop wouldnt work for this column

X = cc_df[['med_sat_value']] #predictor variable
y = cc_df['high_completion'] #target variable

print(f'Number of missing values in X: {X.isna().sum()}') #check for missing values in X
print(f'Number of missing values in y: {y.isna().sum()}') #check for missing values in y
#print(cc_df['high_completion'].value_counts())


# %%
#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#scale X_train and X_test using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Build kNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) #fit model to training data

#Predict on test set
y_pred = knn.predict(X_test) #gives predicted class labels (0 or 1)
y_prob = knn.predict_proba(X_test)[:, 1] #get predicted probabilities for high completion class (1)
# 0.33 → probably low completion ; 0.9 → probably high completion
#print(f'Predicted class labels: {y_pred}')
#print(f'Predicted probabilities for positive class: {y_prob}')


# %%[markdown]
# Step 3: Create DataFrame with results 
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Predicted_Prob': y_prob
})
print(results_df.head())

# %%[markdown]
# Step 4: Question:
# If we adjusted k, a larger k would create a smoother decision boundary. 
#The confusion matrix would not look the same as the same threshhold because 
# the predicted class labels would change if k is adjusted.
# Threshold determines classification cutoff for predicted probabilities. 
# Different probabilities would lead to different classifications and thus a different confusion matrix.

# %%[markdown]
# Step 5: Evaluation
#The confusion matrix shows that the model correctly classified 577 low-completion schools
# and 233 high-completion institutions. However, it incorrectly classified 282 high-completion schools as low-completion.
#This suggests that median SAT score provides some predictive value, but it is not sufficient on its own to accurately 
# classify institutional graduation outcomes. Furthermore, the model appears to be better at identifying low-completion schools 
# than high-completion schools.

# %%
cm = confusion_matrix(y_test, y_pred)   
print(f' Confusion Matrix: {cm}')

accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
print(f' Accuracy: {accuracy:.2f}')

# %%[markdown]
# Step 6: Functions for (1) data cleaning and splitting into training|testing (2)train and test the model with different k and threshold values
# %% Define function for data cleaning and splitting into training/testing sets
def clean_split(data, target_col):
    data = data.copy() #avoid modifying original data
    
    #Create binary target variable based on graduation rate
    data['binary_target'] = (
        data[target_col] > data[target_col].median()
    ).astype(int)

    #Fill missing values in predictor and target variable with median
    columns = ['med_sat_value', 'binary_target']      
    for col in columns:
        data[col].fillna(data[col].median(), inplace=True)
    cc_df.fillna({'med_sat_value': cc_df['med_sat_value'].median()}, inplace=True) #fill missing values in med_sat_value with median. for loop wouldnt work for this column
    
    #Define X and y
    X = data[['med_sat_value']]
    y = data['binary_target']

    #Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Scale X_train and X_test using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  

    return X_train, X_test, y_train, y_test

# %% Define function to train and test kNN model with different k and threshold values
def train_test_knn(X_train, X_test, y_train, y_test, k, threshold):
    #Build kNN model with specified k
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train) #fit model to training data

    #Predict on test set
    probabilities = model.predict_proba(X_test)[:, 1] #get predicted probabilities for high completion class (1)
    predictions = [1 if p > threshold else 0 for p in probabilities] #classify based on threshold

    #Calculate confusion matrix and accuracy
    cm = confusion_matrix(y_test, predictions)
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
    print('k = ', k, 'threshold = ', threshold, 'accuracy = ', accuracy) 
    print('Confusion Matrix:\n', cm)

    return cm, accuracy   
    
#%%
#Test multiple k and threshold values
X_train, X_test, y_train, y_test = clean_split(cc_df, 'grad_100_value')

for k in [3, 5, 7]:
    for threshold in [0.4, 0.5, 0.6]:
        train_test_knn(X_train, X_test, y_train, y_test, k, threshold)

# %%[markdown]
# Step 7: Performance?
#The model performed well, with higheset accuracy of about 72% when k=7 and threshold=0.4.
#Increasing k slightly improved predictive values. Lowering the threshold increased the identifying 
# high-completion schools, while raising it increase false negatives and reduced overall accuracy.
#Although changing k and threshold values had some postive impact on performance, the overall accuracy
# of the model suggests that median SAT scores alone is not a strong predictor of institutional graduation rates.
 

# %%[markdown]
# Step 8: Another variable as target.
X_train, X_test, y_train, y_test = clean_split(cc_df, 'aid_value')
train_test_knn(X_train, X_test, y_train, y_test, k=7, threshold=0.4)



# %%

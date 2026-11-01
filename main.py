import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Reading the training data (has answers) and test data(to predict has no answers)
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Handling missing value and replacing them with the median of the respective ones
fare_median = train_df['Fare'].median() 
embarked_mode = train_df['Embarked'].mode()[0]

# Fill in the missing values with the values found above
for df in [train_df, test_df]:
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)


# Extracting titles such as Mr and Mrs, also combining unique titles that does not have enough volume to one group called rare.
for df in [train_df, test_df]:
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Cap', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss') # In french miss is called mlle and ms we are just combining them to miss so that the model wont have any confusion.
    df['Title'] = df['Title'].replace('Mme', 'Mrs') # same as above

title_age_median = train_df.groupby('Title')['Age'].median()
for df in [train_df, test_df]:
    df['Age'] = df['Age'].fillna(df['Title'].map(title_age_median))

# Counting if the person is in a family or solo and if the family is big or small just counting numbers
for df in [train_df, test_df]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


# Features is just a fancy work for the clues
# Listing the colums we want to use to make our prediction
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize']


# Turning the catogories into like male/female into no like 1s and 0s
# So that the model can work on them
x_train = pd.get_dummies(train_df[features], drop_first=True)
y_train = train_df['Survived'] # The actual answer will be 1 if survived and 0 if not
x_test = pd.get_dummies(test_df[features], drop_first=True) # Giving the same clues for the test too so that the model dont study from the test

x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

# Rescales all numbers to a standard range so that big numbers like fare dosent overpower small numbers like age
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Tests on the model of 5 equal slices of train data to check accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
scores = cross_val_score(log_reg, x_train_scaled, y_train, cv=cv, scoring='accuracy')

# Trains the model on all training data then predicts who survived in the test data
log_reg.fit(x_train_scaled, y_train)
test_predictions = log_reg.predict(x_test_scaled)

# Creating the submission data frame
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

# Saving the submission data to csv format
submission.to_csv('submission-lr2.csv, index=False')

# Printing feedback to the terminal
print(f"Average CV Accuracy: {scores.mean() * 100:.2f}%")
print(f"Predicted Survived: {sum(test_predictions)} | Predicted Deceased: {len(test_predictions) - sum(test_predictions)}")
print("First 5 Predictions")
print(submission.head())
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Loading the data!
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Handling missing value and replacing them with the median of the respective ones
age_median = train_df['Age'].median()
fare_median = train_df['Fare'].median()
embarked_mode = train_df['Embarked'].mode()[0]

# Fill in the missing values with the values found above
for df in [train_df, test_df]:
    df['Age'] = df['Age'].fillna(age_median)
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)


# Features is just a fancy work for the clues we give
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


# Turning the catogories into like male/female into no like 1s and 0s
# So that the model can work on them
x_train = pd.get_dummies(train_df[features], drop_first=True)
y_train = train_df['Survived'] # The actual answer will be 1 if survived and 0 if not
x_test = pd.get_dummies(test_df[features], drop_first=True) # Giving the same clues for the test too so that the model dont study from the test

# Rescales all numbers to a standard range so that big numbers like fare dosent overpower small numbers like scale
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


submission.to_csv('submission-lr1.csv, index=False')

# Printing the values
print(f"Average CV Accuracy: {scores.mean() * 100:.2f}%")
print(f"Predicted Survived: {sum(test_predictions)} | Predicted Deceased: {len(test_predictions) - sum(test_predictions)}")
print("First 5 Predictions")
print(submission.head())
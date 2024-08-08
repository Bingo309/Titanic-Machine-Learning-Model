### Titanic Predictions Machine Model

The Model uses logistic regression to predict whether a passenger survived the Titanic disaster based on features such as age, sex, class, and other related attributes after preprocessing and encoding the necessary data.

## 1. Fill missing values in the "Age" column
```python
data['Age'].fillna(data['Age'].mean(), inplace=True)
```

![image](https://github.com/user-attachments/assets/569759a7-a8e0-4204-b5cb-0352ef9431d8)
Replaces missing values in the "Age" column with the average age.



## 2. Fill missing values in the "Embarked" column

```python
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
```

![image](https://github.com/user-attachments/assets/e3075719-2059-4341-998d-ba0e57c81a0c)

Replaces missing values in the "Embarked" column with the most common embarkation point.



## 3. Drop the "PassengerId" and "Name" columns
Note: Ticket column also needed to dropped as it does not have a clear relationship to the survival of passengers
```python
data = data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
```
![image](https://github.com/user-attachments/assets/401f464f-b8bd-48c7-a952-a9d49ebd7f9c)




## 4. Check for duplicates in the dataset

```python

duplicates = data.duplicated().sum()
print(duplicates)
```

![image](https://github.com/user-attachments/assets/ff1a8134-bdbe-4015-976e-792806c25878)

Counts and prints the number of duplicate rows in the dataset.



## 5. Drop duplicate rows***

```python
data = data.drop_duplicates()
```

![image](https://github.com/user-attachments/assets/b9ea34c6-0e54-4df3-bfcf-b9df81eb72af)

Removes duplicate rows to ensure each entry is unique.



## 6. Split the data into training and testing sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

![image](https://github.com/user-attachments/assets/ea2ae2c2-4e19-4406-b920-1cc05343eaea)

Divides the data into training and testing sets for model evaluation.



## 7. Create and train a Logistic Regression model

```python
from sklearn.linear_model import LogisticRegression


model = LogisticRegression(max_iter=1000)


model.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/be5e964e-096f-4cc1-93eb-de209e414078)

Initializes and trains a Logistic Regression model using the training data.



## 8. Predict and evaluate model accuracy

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the model: {accuracy:.2f}")
```

![image](https://github.com/user-attachments/assets/e3da420e-c3ce-416d-bfd6-020d979a389e)

and finally we get the Accuracy of the model which is [0.79]


import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
conn = sqlite3.connect('coverageF.db')

with open('coverageF.sql', 'r') as sql_file:
    conn.executescript(sql_file.read())

df = pd.read_sql('SELECT * FROM table_name', con=conn)

# Prepare the data
X = df[['feature_1', 'feature_2']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Close the connection
conn.close()

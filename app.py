from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("Customer-Churn-Records.csv")

# Feature selection
X = data.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])
y = data['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Card Type']

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Combine preprocessing and model into a single pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train the Random Forest model with preprocessing
pipeline.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Create a DataFrame from the form data
    input_data = pd.DataFrame(data, index=[0])
    
    # Ensure column order matches the order used during model training
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    # Predict churn
    prediction = pipeline.predict(input_data)[0]
    
    if prediction == 1:
        result = "Customer is likely to churn."
    else:
        result = "Customer is not likely to churn."
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

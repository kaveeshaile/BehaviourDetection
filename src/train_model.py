import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

file_path = 'data/elder_data.csv'

def data_ingestion(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully")
        return data
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"No data: {e}")
    except Exception as e:
        print(f"Error reading the data: {e}")

def transformation(data):
    try:
        X = data.drop('label', axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data transformed successfully")
        return X_train, X_test, y_train, y_test
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"Error during transformation: {e}")

def train_model(X_train, y_train):
    try:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        print("Model trained successfully")
        return model
    except Exception as e:
        print(f"Error during model training: {e}")

def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model predictions: {predictions}")
        print(f"Model accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        print(f"Error during model evaluation: {e}")

def save_model(model, model_path='model.pkl'):
    try:
        joblib.dump(model, model_path)
        print(f"Model saved successfully at {model_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")

if __name__ == "__main__":
    data = data_ingestion(file_path)
    if data is not None:
        X_train, X_test, y_train, y_test = transformation(data)
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            model = train_model(X_train, y_train)
            if model is not None:
                evaluate_model(model, X_test, y_test)
                save_model(model)
    print('Model process complete')

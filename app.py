import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import spacy

# Load the trained Random Forest model
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Load the list of feature names used during training
with open("X_train.pkl", "rb") as feature_file:
    feature_names = pickle.load(feature_file)

# Load the German language model for named entity recognition (NER)
nlp = spacy.load("de_core_news_sm")

# Streamlit UI
def main():
    st.title("House Price Prediction")
    st.write("Enter the details below to predict house prices for the next 12 months.")

    # User input for features
    limit_price = st.number_input("Limit Price", value=100000, step=1000)
    house_info = st.text_input("House Info", "Details about the house...")

    # Preprocess inputs
    def preprocess_inputs(limit_price, house_info):
        # Extract named entities from house_info
        def extract_entities(text):
            doc = nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]

        house_features = extract_entities(house_info)
        house_features_str = ', '.join([f"{text} ({label})" for text, label in house_features])

        input_data = pd.DataFrame({
            'limit_price': [limit_price],
            'house_features': [house_features_str]
        })

        # Perform one-hot encoding for house_features dynamically
        encoded_features = pd.get_dummies(input_data['house_features'])
        input_data = pd.concat([input_data.drop(columns=['house_features']), encoded_features], axis=1)

        # Align input data with the saved feature names
        # Add missing columns and drop extra columns
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        return input_data

    # Predict prices for the next 12 months
    if st.button("Predict"):
        predictions = []
        for i in range(12):
            input_data = preprocess_inputs(limit_price, house_info)

            # Ensure input data matches model's expected feature size
            input_data = input_data[feature_names]  # Align strictly with training features

            prediction = rf_model.predict(input_data)[0]
            predictions.append(prediction)

        # Display predictions
        st.write("Predicted Prices for the Next 12 Months:")
        months = [f"Month {i+1}" for i in range(12)]
        results = pd.DataFrame({"Month": months, "Predicted Price": predictions})
        st.dataframe(results)

        # Plot the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(months, predictions, marker='o')
        plt.title("House Price Predictions for the Next 12 Months")
        plt.xlabel("Month")
        plt.ylabel("Predicted Price")
        plt.grid(True)
        st.pyplot(plt)

if __name__ == "__main__":
    main()

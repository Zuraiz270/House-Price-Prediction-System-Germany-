
# House Price Prediction System (Germany)

This project is a **House Price Prediction System** designed for the German real estate market. It uses machine learning and named entity recognition (NER) to predict house prices for the next 12 months based on user inputs and house features.

## Files in the Project

- **`rf_model.pkl`**: A trained Random Forest model for predicting house prices.
- **`feature_names.pkl`**: A pickle file containing the feature names used during model training.
- **`app.py`**: The Streamlit application providing the user interface for predictions.
- **`house-price-prediction-germany.ipynb`**: A Jupyter Notebook containing the code for preprocessing, model training, and evaluation.
- **`requirements.txt`**: Contains all the Python dependencies for this project.

## Requirements

- Python 3.10 or higher
- Libraries:
  - Streamlit
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - SpaCy (with `de_core_news_sm` model)

## Installation

1. Clone the repository or download the zip file.
2. Install the required dependencies:
3. Download and install the German SpaCy model:
   ```bash
   python -m spacy download de_core_news_sm
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the application in your browser (usually at `http://localhost:8501`).
3. Enter the following:
   - **Limit Price**: A base price estimate for the house.
   - **House Info**: A description of the house including its features (e.g., location, size, property type).
4. Click on the "Predict" button to generate price predictions for the next 12 months.

## Features

- **Named Entity Recognition (NER)**: Extracts house features such as location and type from the input description using the SpaCy German language model (`de_core_news_sm`).
- **Dynamic One-Hot Encoding**: Converts extracted house features into input data for the trained Random Forest model.
- **12-Month Price Predictions**: Outputs predicted house prices for each of the next 12 months.
- **Visualization**: Provides a line plot of the predicted prices.

## Data Processing and Model Training

The Jupyter Notebook (`house-price-prediction-germany.ipynb`) includes:
1. **Data Preprocessing**:
   - Feature engineering using NER.
   - Handling missing values and encoding features.
2. **Model Training**:
   - Random Forest Regressor for price prediction.
   - One-hot encoding for categorical features.
3. **Evaluation**:
   - Metrics such as Mean Squared Error (MSE) and R-squared.

## Limitations

- The model's predictions are limited to the features and data provided during training.
- Feature extraction relies on correctly formatted input text.

## Acknowledgements

- **Kaggle**: For providing house price datasets used in training.
- **Streamlit**: For the web application framework.
- **SpaCy**: For the natural language processing tools.

## Future Enhancements

- Add support for more advanced NER models to improve feature extraction.
- Expand to include additional features such as market trends and economic factors.
- Integrate database support for saving and retrieving predictions.

---

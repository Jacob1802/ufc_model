# UFC Prediction Model

This project aims to predict outcomes of upcoming UFC fights based on historical fight data. It involves scraping fight data, preprocessing it, and training machine learning models to make predictions. The results are stored in a CSV file for further analysis.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/Jacob1802/ufc_model.git
   cd ufc-prediction-model
    ```
2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    On Linux/Mac:
        source venv/bin/activate
    On Windows:
        venv\Scripts\activate
    ```
3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Run the main script:

```sh
python main.py

This will:

- Scrape the latest UFC event data.
- Prepare the data for modeling.
- Train models and make predictions.
- Save the predictions to `data/predictions.csv`.
```

## Model Details

The project uses a combination of machine learning models to predict various fight outcomes:

- **Linear Regression** for statistical predictions (e.g., strikes landed, takedowns).
- **LightGBM Classifier** for win/loss predictions.

## Data Processing

- **Feature Engineering**: Preprocesses the raw fight data to extract relevant features for modeling.
- **Label Encoding**: Encodes categorical variables for model compatibility.

## Training

- Models are trained using historical fight data (`data/fights.csv`).
- The data is split into training and testing sets to evaluate model performance.

## Prediction

- The trained models predict outcomes for upcoming fights (`data/upcoming_fight_data.csv`).
- Predictions include the winner and confidence levels for each fight.
- Results are saved to `data/predictions.csv`.

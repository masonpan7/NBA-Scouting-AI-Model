# 🏀 NBA Scouting AI Model

This project uses a machine learning model with a FastAPI backend to predict potential NBA breakout players based on historical statistics.

## 📌 How It Works

- **Data Source**: `Seasons_Stats.csv` contains NBA player stats from past seasons.
- **Preprocessing**: The data is cleaned and formatted using `Data_Preprocessing.py`.
- **Model Training**: A `RandomForestClassifier` is trained in `Breakout_Model.py` and saved using `joblib`.
- **Prediction API**: The `app.py` file runs a FastAPI server to handle user input and return predictions.
- **Frontend**: `index.html` + `styles.css` provide a simple form UI that submits data to the API.

## 🛠 Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- HTML/CSS (vanilla)
- Uvicorn (for running the API)

## 🚀 To Run Locally

1. Install dependencies:
    ```
    pip install fastapi uvicorn pandas scikit-learn python-multipart jinja2
    ```

2. Start the API server:
    ```
    uvicorn app:app --reload
    ```

3. Open `index.html` in your browser to interact with the model.

---

Created by [@masonpan7](https://github.com/masonpan7)

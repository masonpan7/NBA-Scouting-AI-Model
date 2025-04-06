🏀 NBA Scouting AI Model
This project leverages machine learning and FastAPI to predict potential breakout NBA players based on historical player statistics. It includes data preprocessing, model training, and a web interface for making predictions.

🚀 Features
Cleans and preprocesses raw player data

Trains a machine learning model to predict breakout potential

Fast, asynchronous API built with FastAPI

Web frontend for easy interaction

Interactive prediction interface styled with HTML/CSS

📁 Project Structure
graphql
Copy
Edit
NBA-Scouting-AI-Model/
├── Breakout_Model.py         # ML model training script
├── Data_Preprocessing.py     # Cleans and preprocesses raw data
├── Seasons_Stats.csv         # Raw NBA player data
├── player_data.csv           # Processed data
├── model.pkl                 # Trained ML model
├── scaler.pkl                # StandardScaler object
├── app.py                    # FastAPI backend
├── index.html                # Frontend UI
├── styles.css                # Web UI styling
⚙️ Installation
bash
Copy
Edit
git clone https://github.com/masonpan7/NBA-Scouting-AI-Model.git
cd NBA-Scouting-AI-Model
pip install -r requirements.txt
uvicorn app:app --reload
Then go to http://127.0.0.1:8000 in your browser.

You can also explore the FastAPI Swagger docs at:

arduino
Copy
Edit
http://127.0.0.1:8000/docs
🧠 How It Works
Data_Preprocessing.py loads and cleans raw NBA player stats.

Breakout_Model.py trains a RandomForestClassifier to identify potential breakout players.

The model and scaler are saved as .pkl files for inference.

app.py hosts the FastAPI server and loads the model to serve predictions.

The frontend form (HTML/CSS) allows users to input player stats and receive a prediction.

📈 Input Features
Points per Game

Minutes Played

Assists

Rebounds

Shooting Percentages (FG, 3PT, FT)

Other key advanced stats

🔮 Prediction Output
The app predicts whether a player is likely to "Break Out" or not in their upcoming season based on input stats.

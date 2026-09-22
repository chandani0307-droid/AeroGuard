# AeroGuard 🛩️

## Real-Time Aircraft Engine Failure Prediction

AeroGuard is a machine learning based application designed to monitor aircraft engine sensor data and predict the Remaining Useful Life (RUL) of an engine.

## Features

- Real-time engine health monitoring
- Remaining Useful Life (RUL) prediction
- Engine health status and risk level
- Sensor data visualization
- Maintenance alerts and recommendations
- Interactive Streamlit dashboard

## Machine Learning Model

The project uses a **Gradient Boosting Regressor** for RUL prediction.

### Input Features

- Engine Cycle
- HPC Outlet Temperature
- LPC Outlet Temperature
- Temperature Difference
- HPC Temperature Rolling Mean

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Jupyter Notebook

## Project Structure

```text
AeroGuard/
│
├── app.py
├── notebook.ipynb
├── gradient_boosting_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── data/
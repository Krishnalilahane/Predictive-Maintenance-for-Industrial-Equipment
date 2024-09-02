# Predictive Maintenance for Industrial Equipment

## Overview
This project aims to predict the Remaining Useful Life (RUL) of industrial equipment using machine learning techniques. The goal is to implement a predictive maintenance strategy that minimizes unplanned downtime, optimizes maintenance schedules, and enhances the overall lifespan of the equipment.

## Project Structure
The repository is organized as follows:

- **data/**: Contains the datasets used for training and testing the models, including:
  - `train_fd001.csv`: Training dataset with sensor measurements and operational settings.
  - `test_fd001.csv`: Test dataset with similar features as the training dataset.
  - `rul_fd001.csv`: Contains the actual RUL values for the test dataset.
  
- **notebooks/**: Jupyter notebooks used for data exploration, feature engineering, model training, and evaluation.
  - `data_exploration.ipynb`: Notebook for exploring and visualizing the dataset.
  - `model_training.ipynb`: Notebook for training different machine learning models and evaluating their performance.
  - `model_evaluation.ipynb`: Notebook for comparing model performance and selecting the best model.
  
- **scripts/**: Python scripts for preprocessing data, training models, and generating predictions.
  - `preprocess_data.py`: Script for cleaning and preparing the data for model training.
  - `train_model.py`: Script for training the machine learning models.
  - `predict_rul.py`: Script for generating RUL predictions on new data.
  
- **dashboard/**: Contains the Power BI dashboard file used for visualizing the model results and providing actionable insights.
  - `Predictive_Maintenance_Dashboard.pbix`: Power BI dashboard that visualizes key metrics, clusters, and predictions.

- **README.md**: Project overview and setup instructions.

## Key Features
- **Data Exploration & Preprocessing:** Detailed data exploration and preprocessing to handle missing values, feature scaling, and feature engineering.
- **Model Training & Evaluation:** Implementation of various machine learning models, including Random Forest, XGBoost, Polynomial Regression, and a Voting Regressor, to predict RUL.
- **Power BI Dashboard:** An interactive dashboard to visualize equipment performance, sensor measurements, operational settings, and the impact of these factors on RUL.
- **Web Application:** A simple Flask web application that allows users to input data and receive RUL predictions in real-time.

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the data:**
   ```bash
   python scripts/preprocess_data.py
   ```

4. **Train the model:**
   ```bash
   python scripts/train_model.py
   ```
   
6. **Explore the Power BI Dashboard:**
   - Open the `PowerBi_Dashboard.pbix` file in Power BI Desktop to explore the visualizations and insights.

## Results
The best-performing model in this project was the Voting Regressor, which achieved a Mean Squared Error (MSE) of 7930.70 on the test set. The Power BI dashboard provides detailed insights into equipment performance, sensor data trends, and operational settings, helping users make informed maintenance decisions.

## Value Proposition
This project adds significant value to industrial settings by:
- **Reducing Unplanned Downtime:** Predicting equipment failures before they occur.
- **Optimizing Maintenance Schedules:** Allowing for condition-based maintenance.
- **Improving Equipment Lifespan:** Identifying optimal operational settings.
- **Enhancing Safety and Compliance:** Ensuring timely maintenance to avoid safety hazards.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


---

Thank you for exploring this project! If you find it helpful, please give it a star ⭐ on GitHub. For any questions or suggestions, feel free to reach out.

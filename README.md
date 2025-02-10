# Predicting Used Car Prices with Linear Regression

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Files](#project-files)
- [Methodology](#methodology)
  - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Feature Importance Analysis](#feature-importance-analysis)
  - [Predictions on New Car Data](#predictions-on-new-car-data)
- [Results and Insights](#results-and-insights)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Google Colab Notebook](#google-colab-notebook)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Overview
This project aims to predict the price of used cars based on various features such as mileage, year, model, brand, and fuel efficiency. The project follows a structured machine learning pipeline that includes:
- Data exploration and preprocessing
- Feature engineering
- Model training using linear regression
- Model evaluation and feature importance analysis
- Predictions for new unseen car data

## Dataset
We used the **Craigslist Cars and Trucks Data** from Kaggle. The dataset is publicly available and contains over 426,000 entries with features including:
- **Price:** The target variable representing the car price.
- **Manufacturer, Model:** Car brand and model details.
- **Odometer Reading:** Mileage information.
- **Fuel Type & Transmission Type:** Information about the fuel and transmission systems.
  
Access the dataset here:  
[Kaggle's "Used Cars" Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

## Project Files
This repository contains the following files:
- `1_Linear_Regression.ipynb`: Jupyter Notebook containing code for data preprocessing, model training, and evaluation.
- `1_Linear_Regression_report.pdf`: Report summarizing the project's findings and analysis.
- `01. Linear Regression.pdf`: Instructions outlining the tasks and methodology followed in this project.

## Methodology

### Data Exploration and Preprocessing
- **Data Loading:** Loaded and examined the dataset to understand its structure.
- **Missing Values and Outliers:** Identified and handled missing values and outliers.
- **Visualization:** Explored the distribution of key features and the target variable (car price).

### Feature Engineering
- **Feature Extraction:** Derived useful features such as car age from the manufacturing year.
- **Encoding:** Applied label encoding to categorical variables.
- **Normalization and Scaling:** Normalized and scaled numerical features to improve model performance.

### Model Training
- **Data Splitting:** Divided the dataset into training and testing sets.
- **Model Implementation:** 
  - Trained a linear regression model using scikit-learn.
  - Implemented a custom linear regression model as part of coursework.

### Model Evaluation
- **Evaluation Metrics:** Assessed model performance using:
  - **Mean Squared Error (MSE):** 63,529,711.82
  - **R-squared Score:** 0.63
- **Visualization:** Generated scatter plots to compare predicted vs. actual car prices.

### Feature Importance Analysis
- **Key Influencers:** Identified factors influencing car prices:
  - **Manufacturer:** Luxury brands (e.g., Ferrari, Aston-Martin, Tesla) showed a high positive impact.
  - **Fuel Type:** Diesel demonstrated a notable influence.
  - **Odometer Readings:** Had a moderate impact on pricing.

### Predictions on New Car Data
- **Testing:** Evaluated the model on unseen car data.
- **Example Prediction:**
  - **Input:** Manufacturer: Toyota, Model: Corolla, Year: 2018, Odometer: 40,000 miles, Fuel Type: Gasoline, Transmission: Automatic
  - **Predicted Price:** $26,067.73

## Results and Insights
- The model performs well for lower-priced cars (<$50,000) but struggles with luxury vehicles due to higher price variability.
- Key insights include the importance of mileage, fuel efficiency, and brand value in influencing car prices.
- The model shows potential for use in automated pricing systems in used car marketplaces.

## Limitations and Future Improvements
- **Dataset Biases:** The model is limited by inherent biases and feature availability in the dataset.
- **Advanced Techniques:** Incorporating advanced regression techniques such as polynomial regression or regularized regression (Ridge, Lasso) could improve accuracy.
- **Real-Time Data:** Integrating real-time car pricing data may enhance predictive performance.

## Technologies Used
- **Python**
- **Pandas & NumPy:** Data processing and manipulation
- **Scikit-learn:** Machine learning implementation
- **Matplotlib & Seaborn:** Data visualization
- **Jupyter Notebook / Google Colab:** Interactive coding and experimentation

## How to Run the Project Locally
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/used-car-linear-regression.git
   cd used-car-linear-regression
   ```
2. **Install the Required Libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open the `1_Linear_Regression.ipynb` notebook and run the cells sequentially.

## Google Colab Notebook
Alternatively, you can run the project on Google Colab using the following link:  
[Google Colab Notebook](https://colab.research.google.com/drive/1qFJ2K9AnYC2Lpx2soBO-rm6nZYYXFmcY?usp=sharing)

## Contributors
- **Douadjia Abdelkarim**  
  Master 1 Artificial Intelligence, Djilali Bounaama University of Khemis Miliana

## Acknowledgments
- **Kaggle:** For providing the dataset.
- **Scikit-learn:** For the machine learning tools.
- **Djilali Bounaama University:** For academic support.

---
This project is part of coursework on **Machine Learning with Linear Regression** and aims to provide hands-on experience in predictive modeling.

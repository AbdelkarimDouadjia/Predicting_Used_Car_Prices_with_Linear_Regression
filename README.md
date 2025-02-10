# Predicting Used Car Prices with Linear Regression

## Project Overview
This project aims to predict the price of used cars based on various features such as mileage, year, model, brand, and fuel efficiency. The dataset used for this project is publicly available on Kaggle and contains over 426,000 entries with features like price, manufacturer, model, odometer reading, fuel type, and transmission type.

The project follows a structured machine learning pipeline that includes:
- Data exploration and preprocessing
- Feature engineering
- Model training using linear regression
- Model evaluation and feature importance analysis
- Predictions for new unseen car data

## Dataset
We used the **Craigslist Cars and Trucks Data** from Kaggle. You can find the dataset at the following link:
[Kaggle's "Used Cars" Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

## Project Files
This repository contains the following files:
- `1_Linear_Regression.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `1_Linear_Regression_report.pdf`: Report summarizing the project's findings and analysis.
- `01. Linear Regression.pdf`: Instructions outlining the tasks and methodology followed in this project.

## Methodology

### 1. Data Exploration and Preprocessing
- Loaded and examined the dataset to understand its structure.
- Identified and handled missing values and outliers.
- Visualized the distribution of key features and the target variable (car price).

### 2. Feature Engineering
- Extracted useful features, such as car age from the manufacturing year.
- Encoded categorical variables using label encoding.
- Normalized and scaled numerical features to improve model performance.

### 3. Model Training
- Split the dataset into training and testing sets.
- Trained a **linear regression model** using **scikit-learn**.
- Also implemented a custom linear regression model as part of coursework.

### 4. Model Evaluation
- Used **Mean Squared Error (MSE)** and **R-squared score** to evaluate model performance:
  - **MSE:** 63,529,711.82
  - **R-squared Score:** 0.63
- Generated scatter plots to visualize predicted vs. actual car prices.

### 5. Feature Importance Analysis
- Identified key factors influencing car prices:
  - **Manufacturer:** Luxury brands like Ferrari, Aston-Martin, and Tesla had the highest positive impact on price.
  - **Fuel type:** Diesel had a notable influence.
  - **Odometer readings:** Had a moderate impact on price.

### 6. Predictions on New Car Data
- Tested the model with unseen car data:
  - **Example:**
    - Manufacturer: Toyota
    - Model: Corolla
    - Year: 2018
    - Odometer: 40,000 miles
    - Fuel Type: Gasoline
    - Transmission: Automatic
  - **Predicted Price:** $26,067.73

## Results and Insights
- The model performs well for lower-priced cars (<$50,000) but struggles with luxury vehicles due to price variability.
- Buyers should consider mileage and fuel efficiency, while sellers can leverage premium brand value for higher pricing.
- The model can be used in **automated pricing systems** for used car marketplaces.

## Limitations and Future Improvements
- The model is limited by dataset biases and feature availability.
- Advanced regression techniques such as **polynomial regression** or **regularized regression (Ridge, Lasso)** could improve accuracy.
- Incorporating real-time car pricing data could enhance predictive performance.

## Technologies Used
- **Python**
- **Pandas & NumPy:** Data processing and manipulation
- **Scikit-learn:** Machine learning implementation
- **Matplotlib & Seaborn:** Data visualization
- **Jupyter Notebook / Google Colab:** Interactive coding and experimentation

## How to Run the Project Locally
To run the project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/used-car-linear-regression.git
   cd used-car-linear-regression
    ```
2. **Install the Required Libraries:** 
   Make sure you have installed all the necessary libraries to run the Jupyter Notebook. You can install them using the following command:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. **Launch Jupyter Notebook:**
    Run the following command to open the Jupyter Notebook in your browser:
    ```bash
    jupyter notebook
    ```
    Open the `1_Linear_Regression.ipynb` notebook and run the cells to execute the project.

## Google Colab Notebook
You can access the project notebook on Google Colab via the following link:
[Google Colab Notebook](https://colab.research.google.com/drive/1qFJ2K9AnYC2Lpx2soBO-rm6nZYYXFmcY?usp=sharing)

## Contributors
- **Douadjia Abdelkarim** - Master 1 Artificial Intelligence, Djilali Bounaama University of Khemis Miliana

## Acknowledgments
- Kaggle for providing the dataset
- Scikit-learn for machine learning tools
- Djilali Bounaama University for academic support

---

This project is part of coursework on **Machine Learning with Linear Regression** and aims to provide hands-on experience in predictive modeling.


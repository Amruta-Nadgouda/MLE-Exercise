# MLE-Exercise - BMI Prediction

## Problem Statement:
BMI plays a crucial role in determining the insurance price quote. The goal of this project is to implement an application so that customers get a quote instantly and this greatly enhances the public perception of the industry.

Detailed documentation can be found in "MLE_Exercise_Prudential.pdf".

## Approach :
Applying machine learing tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and model testing to build a solution that should able to predict the premium of the personal for health insurance.

- Data Exploration : Exploring the dataset using pandas, numpy, matplotlib, and seaborn.

- Exploratory Data Analysis : Plotted different graphs to get more insights about dependent and independent variables/features.

- Feature Engineering : Numerical features scaled down, Categorical features encoded and Polynomial Features are created.

- Model Building : In this step, first split the data into train and test sets. After that model is trained on different Machine Learning Algorithms such as:
  - Linear Regression
  - Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Light GBM Regressor

- Model Selection : Tested all the models to check the R2Score, MAE and RMSE.

- Pickle File : Selected model as per best R2Score, MAE and RMSE and created pickle file using pickle library.

## Webpage & Deployment : 
Created a web application that takes all the necessary inputs from the user & shows the output. Then deployed project on the Heroku Platform.
Deployment Link :
https://predictbmi.herokuapp.com/

## Libraries used :
1) Pandas
2) Numpy
3) Matplotlib, Seaborn
4) Scikit-Learn
5) Flask
6) HTML

## Technical Aspects :
1) Python 
2) Front-end : HTML
3) Back-end : Flask
4) Deployment : Heruko

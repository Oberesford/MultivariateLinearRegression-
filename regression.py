from sklearn import linear_model
import pandas as pd
import numpy as np


#Reading in our data
df = pd.read_csv("Life Expectancy Data.csv", usecols=["LifeExpectancy","Adult Mortality","infant deaths","Alcohol"]).fillna(0)
#Writing a cleaned data set as some of our data values are missing
df.to_csv("cleaned.csv")


#Model fitting
reg = linear_model.LinearRegression()
# Taking θ1, θ2, θ3 and our hθ(x)
reg.fit(df[["Alcohol","Adult Mortality","infant deaths"]], df.LifeExpectancy)

reg.coef_ # Coefficients for our Xn values => hθ(x) = θTx 
reg.intercept_ # y Intercept

#Data we want to test (θ1, θ2, θ3)
var = [10,9.92,3.9] # For the UK 
# Prediction
print(f"Prediction: {reg.predict([var])}")
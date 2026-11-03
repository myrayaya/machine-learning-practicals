import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load the data
df = pd.read_csv(r"C:/programs/machine-learning-practicals/ml_prac_1/FAOSTAT_data_en_11-3-2026.csv")
df = df.pivot_table(index = "Year", columns = "Element", values = "Value")
df = df.reset_index()

df = df[["Year", "Area harvested", "Yield", "Production"]]
df = df.dropna()
df = df.sort_values("Year")

print("number of years: ", len(df))
print(df.head())

# train-test split
split_point = int(len(df) * 0.8)
train = df.iloc[:split_point]
test = df.iloc[split_point:]

print("Training years: ", train["Year"].min(), "to", train["Year"].max())
print("Testing years: ", test["Year"].min(), "to", test["Year"].max())

y_train = train["Production"]
y_test = test["Production"]

# linear regression
X_train = train[["Year"]]
X_test = test[["Year"]]

model1 = LinearRegression()
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

mae1 = mean_absolute_error(y_test, pred1)
rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
r2_1 = r2_score(y_test, pred1)

print("Mean Absolute Error: ", round(mae1, 2))
print("Root Mean Squared Error: ", round(rmse1, 2))
print("R-squared: ", round(r2_1, 4))

# polynomial regression
poly = PolynomialFeatures(degree = 2, include_bias = False)
year_mean = train["Year"].mean()
X_train_poly = poly.fit_transform(train[["Year"]] - year_mean)
X_test_poly = poly.transform(test[["Year"]] - year_mean)

model2 = LinearRegression()
model2.fit(X_train_poly, y_train)

pred2 = model2.predict(X_test_poly)

mae2 = mean_absolute_error(y_test, pred2)
rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
r2_2 = r2_score(y_test, pred2)

print("Mean Absolute Error: ", round(mae2, 2))
print("Root Mean Squared Error: ", round(rmse2, 2))
print("R-squared: ", round(r2_2, 4))

# multivariate regression
features = ["Area harvested", "Yield"]
X_train_multi = train[features]
X_test_multi = test[features]

model3 = LinearRegression()
model3.fit(np.log(train[features]), np.log(y_train))

pred3 = np.exp(model3.predict(np.log(test[features])))

mae3 = mean_absolute_error(y_test, pred3)
rmse3 = np.sqrt(mean_squared_error(y_test, pred3))
r2_3 = r2_score(y_test, pred3)

print("Mean Absolute Error: ", round(mae3, 2))
print("Root Mean Squared Error: ", round(rmse3, 2))
print("R-Squared: ", round(r2_3, 2))

# comparison
results = pd.DataFrame({"Model" : ["Linear", "Polynomial", "Multivariate"], "MAE" : [mae1, mae2, mae3], "RMSE" : [rmse1, rmse2, rmse3], "R2" : [r2_1, r2_2, r2_3]})
print(results.round(4).to_string(index = False))

best = results.loc[results["RMSE"].idxmin(), "Model"]
print("Best model (lowest RMSE): ", best)

# plotting the results
plt.figure(figsize = (10, 6))
plt.plot(df["Year"], df["Production"], "k-o", label = "Actual")
plt.plot(test["Year"], pred1, "--", label = "Linear Regression")
plt.plot(test["Year"], pred2, "--", label = "Polynomial Regression")
plt.plot(test["Year"], pred3, "--", label = "Multivariate Regression")
plt.xlabel("Year")
plt.ylabel("Production")
plt.title("Actual vs Predicted (test years)")
plt.legend()
plt.show()
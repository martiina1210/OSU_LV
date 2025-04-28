import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score



data = pd.read_csv('data_C02_emission.csv')
x = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)' ]]
y = data['CO2 Emissions (g/km)']

print(data)
print(y)

print(x)


x_train , x_test , y_train , y_test = train_test_split (x , y , test_size = 0.2 , random_state =1 )


plt.scatter(x_train['Fuel Consumption Comb (L/100km)'], y_train, linewidth=1, marker=".", color="blue")
plt.scatter(x_test['Fuel Consumption Comb (L/100km)'], y_test, linewidth=1, marker=".", color="red")
plt.xlabel("Fuel Consumption Comb (L/100km)")
plt.ylabel("Emissions")
plt.show()




sc = MinMaxScaler()
x_train_n = sc.fit_transform(x_train)
x_test_n = sc.transform(x_test)


plt.hist(x_train['Fuel Consumption City (L/100km)'])
plt.show()


plt.hist(x_train_n[::, 2])
plt.show()


linearModel = lm.LinearRegression()
linearModel.fit(x_train_n, y_train )
print(linearModel.coef_)

y_test_p = linearModel.predict(x_test_n)
plt.scatter(y_test, y_test_p)
plt.show()

MSE = mean_squared_error(y_test, y_test_p)
RMSE = root_mean_squared_error(y_test, y_test_p)
MAE = mean_absolute_error(y_test, y_test_p)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2 = r2_score(y_test , y_test_p)

print(MSE)
print(RMSE)
print(MAE)
print(MAPE)
print(R2)

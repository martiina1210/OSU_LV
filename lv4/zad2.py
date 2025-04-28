import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn . preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

cars = pd.read_csv('data_C02_emission.csv')
x = cars[['Engine Size (L)', 'Fuel Type', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)' ]]
y = cars['CO2 Emissions (g/km)']

ohe = OneHotEncoder(sparse_output=False)
x_encoded = ohe.fit_transform(x[['Fuel Type']])
encoded_columns = ohe.get_feature_names_out(['Fuel Type'])
x_encoded_df = pd.DataFrame(x_encoded, columns=encoded_columns)
x = pd.concat([x.drop(columns=['Fuel Type']), x_encoded_df], axis=1)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state =1 )


linearModel = lm.LinearRegression()
linearModel.fit(x_train, y_train)
print(linearModel.coef_)

y_test_p = linearModel.predict(x_test)
plt.scatter(y_test, y_test_p)
plt.show()

MAE = mean_absolute_error(y_test , y_test_p)
MAPE = mean_absolute_percentage_error(y_test , y_test_p)
MSE = mean_squared_error(y_test , y_test_p)
RMSE = root_mean_squared_error(y_test , y_test_p)
R2 = r2_score(y_test , y_test_p)

print(MAE, MAPE, MSE, RMSE, R2)

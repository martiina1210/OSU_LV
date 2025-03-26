import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("osu_lv/LV3/data_C02_emission.csv")

plt.figure()

# a

data['CO2 Emissions (g/km)'].plot(kind='hist', bins = 20)
plt.show()

# b

data['Fuel Type'] = data['Fuel Type'].astype("category")
data.plot.scatter(
    x='Fuel Consumption City (L/100km)',
    y='CO2 Emissions (g/km)',
    c='Fuel Type',
    cmap="magma"
)
plt.show()

# c

data.boxplot(column = ['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()

# d

plt.figure()
groupedby_fuel_type = data.groupby(by='Fuel Type')['Make'].count()
groupedby_fuel_type.plot(kind="bar")
plt.show()

# e

plt.figure()
groupedby_cylinders = data.groupby(by='Cylinders')['CO2 Emissions (g/km)'].mean()
groupedby_cylinders.plot(kind="bar")
plt.show()

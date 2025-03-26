import pandas as pd

data = pd.read_csv('osu_lv/LV3/data_C02_emission.csv')

# a

print("Koliko mjerenja sadrzi DataFrame:", len(data))
print("\n")
print("Kojeg je tipa svaka velicina?\n", data.dtypes)
print("\n")
print("Postoje li izostale vrijednosti?\n", data.isnull().sum())
data.dropna(axis=0)

print("Postoje li duplicirane vrijednosti?\n", data.duplicated().sum())
data.drop_duplicates()

data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']] = data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']].astype("category")
print(data.dtypes)

print("\n")

# b 

highestFuelCity = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(3)
lowestFuelCity = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=True).head(3)

print("Automobili sa najvećom gradskom potrošnjom goriva:")
for index, row in highestFuelCity.iterrows():
    print(f"Proizvođač: {row['Make']}, Model: {row['Model']}, Gradska potrošnja: {row['Fuel Consumption City (L/100km)']} L/100km")

print("\n")

print("\nAutomobili sa najmanjom gradskom potrošnjom goriva:")
for index, row in lowestFuelCity.iterrows():
    print(f"Proizvođač: {row['Make']}, Model: {row['Model']}, Gradska potrošnja: {row['Fuel Consumption City (L/100km)']} L/100km")

print("\n")

# c

filtered_cars = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)] # !!!!!!!!!!!!!!

print("Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L:", len(filtered_cars))
print("Kolika je prosjecna C02 emisija plinova za ova vozila?", filtered_cars['CO2 Emissions (g/km)'].mean())

# d

print('\n')
audi_cars = data[data.Make == "Audi"]
print("Koliko mjerenja se odnosi na vozila proizvodaca Audi?", len(audi_cars))
audi_cars_with_four_cylinders = audi_cars[audi_cars['Cylinders'] == 4]
print("Kolika je prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara?", audi_cars_with_four_cylinders['CO2 Emissions (g/km)'].mean())

# e

print('\n')
grouped_cars_cylinders = data.groupby('Cylinders')
print(grouped_cars_cylinders.size())
print("Kolika je prosjecna emisija C02 plinova s obzirom na broj cilindara?\n", grouped_cars_cylinders['CO2 Emissions (g/km)'].mean())

# f

print('\n')
cars_dizel = data[data['Fuel Type'] == 'D']
print("Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel", cars_dizel['Fuel Consumption City (L/100km)'].mean())
print("Medijan dizela:", cars_dizel['Fuel Consumption City (L/100km)'].median())
cars_regural = data[data['Fuel Type'] == 'X']
print("Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste benzin", cars_regural['Fuel Consumption City (L/100km)'].mean())
print("Medijan benzina:", cars_regural['Fuel Consumption City (L/100km)'].median())

# g

cars_with_4_cylinders_and_D = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
print('\n')
print(cars_with_4_cylinders_and_D.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(1))

# h

cars_with_manual = data[data['Transmission'].str.startswith('M')]
print(len(cars_with_manual))

# i

print(data.corr(numeric_only=True)) 

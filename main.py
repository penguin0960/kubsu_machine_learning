import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ALL_COLUMNS = [
    'Record Number',
    'Order',
    'NYC Borough, Block and Lot (BBL)',
    'Co-reported BBL Status',
    'BBLs Co-reported',
    'Reported NYC Building Identification Numbers (BINs)',
    'Property Name',
    'Parent Property Id',
    'Parent Property Name',
    'Street Number',
    'Street Name',
    'Zip Code',
    'Borough',
    'DOF Benchmarking Submission Status',
    'Primary Property Type - Self Selected',
    'List of All Property Use Types at Property',
    'Largest Property Use Type',
    'Largest Property Use Type - Gross Floor Area (ft²)',
    '2nd Largest Property Use Type',
    '2nd Largest Property Use - Gross Floor Area (ft²)',
    '3rd Largest Property Use Type',
    '3rd Largest Property Use Type - Gross Floor Area (ft²)',
    'Year Built',
    'Number of Buildings - Self-reported',
    'Occupancy',
    'Metered Areas (Energy)',
    'Metered Areas  (Water)',
    'ENERGY STAR Score',
    'Site EUI (kBtu/ft²)',
    'Weather Normalized Site EUI (kBtu/ft²)',
    'Weather Normalized Site Electricity Intensity (kWh/ft²)',
    'Weather Normalized Site Natural Gas Intensity (therms/ft²)',
    'Source EUI (kBtu/ft²)',
    'Weather Normalized Source EUI (kBtu/ft²)',
    # 'Fuel Oil #1 Use (kBtu)',
    # 'Fuel Oil #2 Use (kBtu)',
    # 'Fuel Oil #4 Use (kBtu)',
    # 'Fuel Oil #5 & 6 Use (kBtu)',
    'Diesel #2 Use (kBtu)',
    'District Steam Use (kBtu)',
    'District Hot Water Use (kBtu)',
    'District Chilled Water Use (kBtu)',
    'Natural Gas Use (kBtu)',
    'Weather Normalized Site Natural Gas Use (therms)',
    'Electricity Use - Grid Purchase (kBtu)',
    'Weather Normalized Site Electricity (kWh)',
    'Total GHG Emissions (Metric Tons CO2e)',
    'Direct GHG Emissions (Metric Tons CO2e)',
    'Indirect GHG Emissions (Metric Tons CO2e)',
    'DOF Property Floor Area (ft²)',
    'Property GFA - Self-reported (ft²)',
    'Water Use (All Water Sources) (kgal)',
    'Municipally Supplied Potable Water - Indoor Intensity (gal/ft²)',
    # 'Release Date',
    'DEP Provided Water Use (kgal)',
    'Automatic Water Benchmarking Eligible',
    'Reported Water Method'
]
USEFUL_COLUMNS = [
    'Co-reported BBL Status',
    'BBLs Co-reported',
    'Reported NYC Building Identification Numbers (BINs)',
    'Parent Property Id',
    'Parent Property Name',
    'DOF Benchmarking Submission Status',
    'Primary Property Type - Self Selected',
    'List of All Property Use Types at Property',
    'Largest Property Use Type',
    'Largest Property Use Type - Gross Floor Area (ft²)',
    '2nd Largest Property Use Type',
    '2nd Largest Property Use - Gross Floor Area (ft²)',
    '3rd Largest Property Use Type',
    '3rd Largest Property Use Type - Gross Floor Area (ft²)',
    'Year Built',
    'Number of Buildings - Self-reported',
    # 'Occupancy',
    'Metered Areas (Energy)',
    'Metered Areas  (Water)',
    # 'Site EUI (kBtu/ft²)',
    'Weather Normalized Site EUI (kBtu/ft²)',
    'Weather Normalized Site Electricity Intensity (kWh/ft²)',
    'Weather Normalized Site Natural Gas Intensity (therms/ft²)',
    # 'Source EUI (kBtu/ft²)',
    'Weather Normalized Source EUI (kBtu/ft²)',
    'Diesel #2 Use (kBtu)',
    'District Steam Use (kBtu)',
    'District Hot Water Use (kBtu)',
    # 'Natural Gas Use (kBtu)',
    'Weather Normalized Site Natural Gas Use (therms)',
    # 'Electricity Use - Grid Purchase (kBtu)',
    # 'Weather Normalized Site Electricity (kWh)',
    # 'Total GHG Emissions (Metric Tons CO2e)',
    # 'Direct GHG Emissions (Metric Tons CO2e)',
    'Indirect GHG Emissions (Metric Tons CO2e)',
    # 'DOF Property Floor Area (ft²)',
    'Property GFA - Self-reported (ft²)',
    # 'Water Use (All Water Sources) (kgal)',
    # 'Municipally Supplied Potable Water - Indoor Intensity (gal/ft²)',
    'DEP Provided Water Use (kgal)',
    'Automatic Water Benchmarking Eligible',
    'Reported Water Method',
]
TARGET_COLUMNS = [
    'ENERGY STAR Score',
]
FEATURE_COLUMNS = [
    # 'Weather Normalized Site EUI (kBtu/ft²)',
    # 'Weather Normalized Site Electricity Intensity (kWh/ft²)',
    # 'Weather Normalized Site Natural Gas Intensity (therms/ft²)',
    #
    'Occupancy',
    # 'Municipally Supplied Potable Water - Indoor Intensity (gal/ft²)',
    # 'Site EUI (kBtu/ft²)',
    'Source EUI (kBtu/ft²)',
    'Natural Gas Use (kBtu)',
    'Electricity Use - Grid Purchase (kBtu)',
    'Total GHG Emissions (Metric Tons CO2e)',
    'Water Use (All Water Sources) (kgal)',
    # 'Weather Normalized Site Electricity (kWh)',
    # 'Direct GHG Emissions (Metric Tons CO2e)',
    'DOF Property Floor Area (ft²)',
]


model = LinearRegression()

nyc = pd.read_excel(
    'labs_data/nyc_2016.xlsx',
    usecols=FEATURE_COLUMNS+TARGET_COLUMNS,
)
print(nyc.columns.tolist())
nyc = nyc.dropna(how="any")
features = nyc[FEATURE_COLUMNS]
targets = nyc[TARGET_COLUMNS]
print('---Исходные данные для обучения (features)---')
print(features)
print('---Исходные данные для обучения (targets)---')
print(targets)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    targets,
    test_size=0.3,
    random_state=42,
)
print('---Исходные данные для обучения (X_train)---')
print(X_train)
print('---Исходные данные для обучения (y_train)---')
print(y_train)
model.fit(X_train, y_train)
model_pred = model.predict(X_test)
print('---Результат обучения (model_pred)---')
print(model_pred)

print('---Ошибка обучения---')
test_target_data = np.array(y_test[TARGET_COLUMNS[0]])
diff = abs(model_pred - test_target_data)
print(np.mean(diff))


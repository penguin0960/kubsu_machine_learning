import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from labs_data.housing_7_data import FEATURE_COLUMNS, TARGET_COLUMNS, FILENAME

model = LinearRegression()

nyc = pd.read_excel(
    FILENAME,
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
print('---Исходные данные для тестирования (y_test)---')
print(y_test)
print('---Результат обучения (model_pred)---')
print(model_pred)

print('---Ошибка обучения---')
test_target_data = np.array(y_test[TARGET_COLUMNS[0]])
diff = abs(model_pred - test_target_data)
print(np.mean(diff))

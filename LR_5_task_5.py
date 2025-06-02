import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# Завантаження даних
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')
        data.append(items)

data = np.array(data)

# Кодування даних
label_encoders = []
X_encoded = np.empty(data.shape)

for i in range(data.shape[1]):
    if data[0, i].isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розбиття на навчальний/тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Модель
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Прогнозування
y_pred = regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

# Тестова точка
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = []

encoder_index = 0
for item in test_datapoint:
    if item.isdigit():
        test_datapoint_encoded.append(int(item))
    else:
        test_datapoint_encoded.append(
            int(label_encoders[encoder_index].transform([item])[0])
        )
        encoder_index += 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Прогнозування для окремої точки
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))

import pandas as pd
from simple_chalk import chalk
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import prepare_data_one_hot
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('2020_june_mini.csv')

prepare_data_one_hot.prepare_2020_june_data(df)

# one hot encoding
one_hot = pd.get_dummies(df['Город'], drop_first=True, prefix='Город')
df = df.drop('Город', axis=1)
df = df.join(one_hot)

# normalize fields
scale_features_mm = MinMaxScaler()
df[['Возраст', 'exp', 'current_job_exp']] = scale_features_mm.fit_transform(df[['Возраст', 'exp', 'current_job_exp']])

# split the data into train and test set
train, test = train_test_split(df, test_size=0.25, random_state=52, shuffle=True)

# define features and targets
features = 'Зарплата.в.месяц'
targets = df.columns.tolist()
targets.remove('N')
targets.remove('Зарплата.в.месяц')
targets.remove('Изменение.зарплаты.за.12.месяцев')
targets.remove('Специализация')
targets.remove('Еще.студент')
targets.remove('Предметная.область')

# split data into features and targets
y_train = train[features].values
y_test = test[features].values
X_train = train[targets].values
X_test = test[targets].values

mlp = MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, early_stopping=True)

mlp.fit(X_train, y_train)

# mlp regression
print(chalk.green('\ncoefficient of determination R^2 of the prediction'))
print('mlp regressor train score = ', round(mlp.score(X_train, y_train), 2))
print('mlp regressor test score = ', round(mlp.score(X_test, y_test), 2))

print(chalk.green('\nmean squared error'))
print('mlp regressor train score = ', metrics.mean_squared_error(y_train, mlp.predict(X_train), squared=False))
print('mlp test score = ', metrics.mean_squared_error(y_test, mlp.predict(X_test), squared=False))

print(chalk.green('\nmean absolute error'))
print('linear regression train score = ', metrics.mean_absolute_error(y_train, mlp.predict(X_train)))
print('linear regression test score = ', metrics.mean_absolute_error(y_test, mlp.predict(X_test)))

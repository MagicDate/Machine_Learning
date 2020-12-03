# external modules
import pandas as pd
from simple_chalk import chalk
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# internal modules
import prepare_data

# load data from csv file
df = pd.read_csv('2020_june_mini.csv')

# prepare dataset
prepare_data.prepare_2020_june_data(df)

# normalize fields
scale_features_mm = MinMaxScaler()
df[['Возраст', 'exp', 'current_job_exp']] = scale_features_mm.fit_transform(df[['Возраст', 'exp', 'current_job_exp']])

# split the data into train and test set
train, test = train_test_split(df, test_size=0.25, random_state=52, shuffle=True)

# define features and targets
features = 'Зарплата.в.месяц'
targets = ['Город',
           'Пол',
           'Возраст',
           'Должность',
           'exp',
           'current_job_exp',
           'Язык.программирования',
           'Образование',
           'Университет',
           'Уровень.английского',
           'Размер.компании',
           'Тип.компании']

# split data into features and targets
y_train = train[features].values
y_test = test[features].values
X_train = train[targets].values
X_test = test[targets].values

# define data models
lr = linear_model.LinearRegression()
dtr = DecisionTreeRegressor(random_state=1, max_depth=10, min_samples_split=2)
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, early_stopping=True)

# training models
lr.fit(X_train, y_train)
dtr.fit(X_train, y_train)
mlp.fit(X_train, y_train)


# quantifying the quality of predictions

# linear regression
print(chalk.green('\ncoefficient of determination R^2 of the prediction'))
print('linear regression train score = ', round(lr.score(X_train, y_train), 2))
print('linear regression test score = ', round(lr.score(X_test, y_test), 2))

print(chalk.green('\nmean squared error'))
print('linear regression train score = ', metrics.mean_squared_error(y_train, lr.predict(X_train), squared=False))
print('linear regression test score = ', metrics.mean_squared_error(y_test, lr.predict(X_test), squared=False))

print(chalk.green('\nmean absolute error'))
print('linear regression train score = ', metrics.mean_absolute_error(y_train, lr.predict(X_train)))
print('linear regression test score = ', metrics.mean_absolute_error(y_test, lr.predict(X_test)))

# decision tree
print(chalk.green('\ncoefficient of determination R^2 of the prediction'))
print('decision tree regressor train score = ', round(dtr.score(X_train, y_train), 2))
print('decision tree regressor test score = ', round(dtr.score(X_test, y_test), 2))

print(chalk.green('\nmean squared error'))
print('decision tree regressor train score = ',
      metrics.mean_squared_error(y_train, dtr.predict(X_train), squared=False))
print('decision tree regressor test score = ', metrics.mean_squared_error(y_test, dtr.predict(X_test), squared=False))

print(chalk.green('\nmean absolute error'))
print('linear regression train score = ', metrics.mean_absolute_error(y_train, dtr.predict(X_train)))
print('linear regression test score = ', metrics.mean_absolute_error(y_test, dtr.predict(X_test)))

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

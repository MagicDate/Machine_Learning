import pandas as pd
from sklearn import linear_model
import prepare_data


df = pd.read_csv('2020_june_mini.csv')
prepare_data.prepare_2020_june_data(df)

y = df['Зарплата.в.месяц'].values
x = df[['Город',
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
        'Тип.компании']].values

# train values
y_train = y[::2]
x_train = x[::2]

# test values
y_test = y[1::2]
x_test = x[1::2]

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)


predict_value = model.predict([[2, 2, 20, 17, 0, 0, 6, 1, 2, 3, 1, 4]])
# print(predict_value)

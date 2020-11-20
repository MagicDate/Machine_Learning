import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from . import prepare_data

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

dtr = DecisionTreeRegressor(random_state=1, max_depth=10, min_samples_split=2)
dtr.fit(x_train, y_train)


def get_score():
    score = dtr.score(x_test, y_test)
    return score


def get_prediction(city, sex, age, position, exp, current_job_exp, programing_language, education, university,
                   english_level, company_size, company_type):
    predict_value = dtr.predict([[city, sex, age, position, exp, current_job_exp, programing_language,
                                  education, university, english_level, company_size, company_type]])
    return predict_value

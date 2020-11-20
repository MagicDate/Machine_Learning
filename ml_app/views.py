from django.shortcuts import render
from django.http import HttpResponse
from .forms import UserForm
from . import linear_regression
from . import decision_tree_regressor
from . import mlp_regressor


def index(request):
    if request.method == 'POST':
        city = request.POST.get("city")
        sex = request.POST.get("sex")
        age = request.POST.get("age")
        position = request.POST.get("position")
        exp = request.POST.get("exp")
        current_job_exp = request.POST.get("current_job_exp")
        programing_language = request.POST.get("programing_language")
        education = request.POST.get("education")
        university = request.POST.get("university")
        english_level = request.POST.get("english_level")
        company_size = request.POST.get("company_size")
        company_type = request.POST.get("company_type")


        linear_regression.get_prediction(int(city), int(sex), int(age), int(position), float(exp), float(current_job_exp), int(programing_language), int(education), int(university),
                                         int(english_level), int(company_size), int(company_type))

        return HttpResponse('<h2>Количественная Оценка Качества Прогнозов</h2>'
                            'LinearRegression = ' + str(linear_regression.get_score()) + '<br/>'
                                                                                         'DecisionTreeRegressor = ' + str(
            decision_tree_regressor.get_score()) + '<br/>'
                                                   'MLPRegressor = ' + str(mlp_regressor.get_score()) +
                            '<h2>Результаты Прогнозов</h2>'
                            'LinearRegression = ' + str(linear_regression.get_prediction(int(city), int(sex), int(age), int(position), float(exp), float(current_job_exp), int(programing_language), int(education), int(university),
                                         int(english_level), int(company_size), int(company_type))) + '<br/>'
                            'DecisionTreeRegressor = ' + str(decision_tree_regressor.get_prediction(int(city), int(sex), int(age),
                                                                                     int(position), float(exp),
                                                                                     float(current_job_exp),
                                                                                     int(programing_language),
                                                                                     int(education), int(university),
                                                                                     int(english_level),
                                                                                     int(company_size),
                                                                                     int(company_type))) + '<br/>'                                                                          
                            'LinearRegression = ' + str(mlp_regressor.get_prediction(
                                int(city), int(sex), int(age), int(position), float(exp), float(current_job_exp), int(programing_language),
                                int(education), int(university),
                                int(english_level), int(company_size), int(company_type))) + '<br/>'
                            )
    else:
        user_form = UserForm()
        return render(request, "index.html", {"form": user_form})

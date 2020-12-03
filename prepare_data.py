import pandas as pd
# In ['Город', 'Должность', 'Опыт работы', 'Текущий опыт работы', 'Язык программирования', 'Возраст',
# 'Пол', 'Образование', 'Университет', 'Уровень.английского', 'Размер.компании', 'Тип.компании']

# Out ['Зарплата в месяц']

# Prepare Data

# Rows that have null value
# print(df[df['Город'].isnull() == True]) 0
# print(df[df['Зарплата.в.месяц'].isnull() == True]) 0
# print(df[df['Изменение.зарплаты.за.12.месяцев'].isnull() == True]) 0
# print(df[df['Должность'].isnull() == True]) 0
# print(df[df['exp'].isnull() == True]) 0
# print(df[df['current_job_exp'].isnull() == True]) 0
# print(df[df['Язык.программирования'].isnull() == True]) 5715
# print(df[df['Специализация'].isnull() == True]) 10197
# print(df[df['Возраст'].isnull() == True]) 0
# print(df[df['Пол'].isnull() == True]) 0
# print(df[df['Образование'].isnull() == True]) 0
# print(df[df['Университет'].isnull() == True]) 877
# print(df[df['Еще.студент'].isnull() == True]) 0
# print(df[df['Уровень.английского'].isnull() == True]) 0
# print(df[df['Размер.компании'].isnull() == True]) 0
# print(df[df['Тип.компании'].isnull() == True]) 0
# print(df[df['Предметная.область'].isnull() == True]) 0


def prepare_2020_june_data(df):
    # Prepare Город number value
    df['Город'] = df['Город'].replace(['Другой'], 0)
    df['Город'] = df['Город'].replace(['Ужгород'], 1)
    df['Город'] = df['Город'].replace(['Харьков'], 2)
    df['Город'] = df['Город'].replace(['Одесса'], 3)
    df['Город'] = df['Город'].replace(['Днепр'], 4)
    df['Город'] = df['Город'].replace(['Киев'], 5)
    df['Город'] = df['Город'].replace(['Львов'], 6)
    df['Город'] = df['Город'].replace(['Винница'], 7)
    df['Город'] = df['Город'].replace(['Николаев'], 8)
    df['Город'] = df['Город'].replace(['Кропивницкий'], 9)
    df['Город'] = df['Город'].replace(['Луцк'], 10)
    df['Город'] = df['Город'].replace(['Черновцы'], 11)
    df['Город'] = df['Город'].replace(['Ровно'], 12)
    df['Город'] = df['Город'].replace(['Ивано-Франковск'], 13)
    df['Город'] = df['Город'].replace(['Житомир'], 14)
    df['Город'] = df['Город'].replace(['Кривой Рог'], 15)
    df['Город'] = df['Город'].replace(['Хмельницкий'], 16)
    df['Город'] = df['Город'].replace(['Херсон'], 17)
    df['Город'] = df['Город'].replace(['Черкассы'], 18)
    df['Город'] = df['Город'].replace(['Чернигов'], 19)
    df['Город'] = df['Город'].replace(['Полтава'], 20)
    df['Город'] = df['Город'].replace(['Запорожье'], 21)
    df['Город'] = df['Город'].replace(['Сумы'], 22)
    df['Город'] = df['Город'].replace(['Мариуполь'], 23)
    df['Город'] = df['Город'].replace(['Тернополь'], 24)
    # print(df['Город'].unique())

    # Prepare Должность number value
    df['Должность'] = df['Должность'].replace(['System Architect'], 1)
    df['Должность'] = df['Должность'].replace(['Senior Software Engineer'], 2)
    df['Должность'] = df['Должность'].replace(['QA Engineer'], 3)
    df['Должность'] = df['Должность'].replace(['Team/Technical Lead'], 4)
    df['Должность'] = df['Должность'].replace(['HTML Coder'], 5)
    df['Должность'] = df['Должность'].replace(['Support'], 6)
    df['Должность'] = df['Должность'].replace(['Director of Engineering / Program Director'], 7)
    df['Должность'] = df['Должность'].replace(['QA Tech Lead'], 8)
    df['Должность'] = df['Должность'].replace(['DevOps'], 9)
    df['Должность'] = df['Должность'].replace(['Product Manager'], 10)
    df['Должность'] = df['Должность'].replace(['Software Engineer'], 11)
    df['Должность'] = df['Должность'].replace(['Sales manager'], 12)
    df['Должность'] = df['Должность'].replace(['Junior QA engineer'], 13)
    df['Должность'] = df['Должность'].replace(['HR'], 14)
    df['Должность'] = df['Должность'].replace(['Project manager'], 15)
    df['Должность'] = df['Должность'].replace(['Senior Project manager / Program Manager'], 16)
    df['Должность'] = df['Должность'].replace(['Junior Software Engineer'], 17)
    df['Должность'] = df['Должность'].replace(['Marketing'], 18)
    df['Должность'] = df['Должность'].replace(['Data Scientist'], 19)
    df['Должность'] = df['Должность'].replace(['DBA'], 20)
    df['Должность'] = df['Должность'].replace(['Product Owner'], 21)
    df['Должность'] = df['Должность'].replace(['Designer'], 22)
    df['Должность'] = df['Должность'].replace(['Data Engineer'], 23)
    df['Должность'] = df['Должность'].replace(['QA Manager'], 24)
    df['Должность'] = df['Должность'].replace(['Senior QA engineer'], 25)
    df['Должность'] = df['Должность'].replace(['Customer Success'], 26)
    df['Должность'] = df['Должность'].replace(['Business analyst'], 27)
    df['Должность'] = df['Должность'].replace(['SysAdmin'], 28)
    df['Должность'] = df['Должность'].replace(['Security Specialist'], 29)
    df['Должность'] = df['Должность'].replace(['System Analyst'], 30)
    df['Должность'] = df['Должность'].replace(['Research Engineer'], 31)
    df['Должность'] = df['Должность'].replace(['Scrum Master'], 32)
    df['Должность'] = df['Должность'].replace(['BI Engineer'], 33)
    df['Должность'] = df['Должность'].replace(['ERP / CRM'], 34)
    df['Должность'] = df['Должность'].replace(['Data Analyst'], 35)
    df['Должность'] = df['Должность'].replace(['Technical writer'], 36)
    df['Должность'] = df['Должность'].replace(['Copywriter'], 37)
    # print(len(df['Должность'].unique()))

    # Prepare Язык программирования number value
    df['Язык.программирования'] = df['Язык.программирования'].fillna(-1)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Other'], 0)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Інше'], 0)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['C#/.NET'], 1)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Python'], 2)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Kotlin'], 3)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Java'], 4)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['PHP'], 5)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['JavaScript'], 6)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['TypeScript'], 7)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Clojure'], 8)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Objective-C'], 9)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['1С'], 10)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['C'], 11)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Golang'], 12)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Swift'], 13)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Scala'], 14)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['C++'], 15)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Erlang'], 16)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Elixir'], 17)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Rust'], 18)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['SQL'], 19)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Delphi'], 20)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Dart'], 21)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['APL'], 22)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['ABAP'], 23)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Haskell'], 24)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Perl'], 25)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Groovy'], 26)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['R'], 27)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Assembler'], 28)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Flex/Flash/AIR'], 29)
    df['Язык.программирования'] = df['Язык.программирования'].replace(['Ruby/Rails'], 30)
    # print(len(df['Язык.программирования'].unique()))

    # Prepare Пол number value
    df['Пол'] = df['Пол'].replace(['женский'], 1)
    df['Пол'] = df['Пол'].replace(['мужской'], 2)
    # print(len(df['Пол'].unique()))

    # Prepare Образование number value
    df['Образование'] = df['Образование'].replace(['Еще студент'], 1)
    df['Образование'] = df['Образование'].replace(['Высшее'], 2)
    df['Образование'] = df['Образование'].replace(['Два высших'], 3)
    df['Образование'] = df['Образование'].replace(['Незаконченное высшее'], 4)
    df['Образование'] = df['Образование'].replace(['Кандидат'], 5)
    df['Образование'] = df['Образование'].replace(['Техникум / Колледж'], 6)
    df['Образование'] = df['Образование'].replace(['Среднее'], 7)
    # print(df['Образование'].unique())

    # Prepare Университет number value
    df['Университет'] = df['Университет'].fillna(-1)
    df['Университет'] = df['Университет'].replace(['Другой вуз'], 0)
    df['Университет'] = df['Университет'].replace(['КНУСА'], 1)
    df['Университет'] = df['Университет'].replace(['ХНУРЭ'], 2)
    df['Университет'] = df['Университет'].replace(['ОНАС'], 3)
    df['Университет'] = df['Университет'].replace(['КНУ им. Шевченко'], 4)
    df['Университет'] = df['Университет'].replace(['ЛНУ им. Франко'], 5)
    df['Университет'] = df['Университет'].replace(['Львовская Политехника'], 6)
    df['Университет'] = df['Университет'].replace(['НТУУ «КПИ»'], 7)
    df['Университет'] = df['Университет'].replace(['ВНУ им. Даля'], 8)
    df['Университет'] = df['Университет'].replace(['ХНУ им. Каразина'], 9)
    df['Университет'] = df['Университет'].replace(['ДонНУ им. Васыля Стуса'], 10)
    df['Университет'] = df['Университет'].replace(['ДонНТУ'], 11)
    df['Университет'] = df['Университет'].replace(['ЗНТУ'], 12)
    df['Университет'] = df['Университет'].replace(['ЧНУ им. Федьковича'], 13)
    df['Университет'] = df['Университет'].replace(['ВНТУ'], 14)
    df['Университет'] = df['Университет'].replace(['НТУ «ХПИ»'], 15)
    df['Университет'] = df['Университет'].replace(['ЧГУ им. Петра Могилы'], 16)
    df['Университет'] = df['Университет'].replace(['ОНУ им. Мечникова'], 17)
    df['Университет'] = df['Университет'].replace(['КНЭУ'], 18)
    df['Университет'] = df['Университет'].replace(['ОНПУ'], 19)
    df['Университет'] = df['Университет'].replace(['КНУТД'], 20)
    df['Университет'] = df['Университет'].replace(['ДНУ им. Гончара'], 21)
    df['Университет'] = df['Университет'].replace(['ЖГТУ'], 22)
    df['Университет'] = df['Университет'].replace(['ЧНТУ'], 23)
    df['Университет'] = df['Университет'].replace(['ДНУЖТ'], 24)
    df['Университет'] = df['Университет'].replace(['НУПТ'], 25)
    df['Университет'] = df['Университет'].replace(['НГУ [Днепропетровск]'], 26)
    df['Университет'] = df['Университет'].replace(['ХНУ [г. Хмельницкий]'], 27)
    df['Университет'] = df['Университет'].replace(['ХАИ'], 28)
    df['Университет'] = df['Университет'].replace(['ХНЭУ'], 29)
    df['Университет'] = df['Университет'].replace(['ГУТ'], 30)
    df['Университет'] = df['Университет'].replace(['УКУ'], 31)
    df['Университет'] = df['Университет'].replace(['НаУКМА'], 32)
    df['Университет'] = df['Университет'].replace(['НметАУ'], 33)
    df['Университет'] = df['Университет'].replace(['СумГУ'], 34)
    df['Университет'] = df['Университет'].replace(['НАУ'], 35)
    df['Университет'] = df['Университет'].replace(['НУК'], 36)
    # print(len(df['Университет'].unique()))

    # Prepare Уровень английского number value
    df['Уровень.английского'] = df['Уровень.английского'].replace(['элементарный'], 1)
    df['Уровень.английского'] = df['Уровень.английского'].replace(['ниже среднего'], 2)
    df['Уровень.английского'] = df['Уровень.английского'].replace(['средний'], 3)
    df['Уровень.английского'] = df['Уровень.английского'].replace(['выше среднего'], 4)
    df['Уровень.английского'] = df['Уровень.английского'].replace(['продвинутый'], 5)
    # print(df['Уровень.английского'].unique())

    # Prepare Размер компании number value
    df['Размер.компании'] = df['Размер.компании'].replace(['до 10 человек'], 1)
    df['Размер.компании'] = df['Размер.компании'].replace(['до 50 человек'], 2)
    df['Размер.компании'] = df['Размер.компании'].replace(['до 200 человек'], 3)
    df['Размер.компании'] = df['Размер.компании'].replace(['до 1000 человек'], 4)
    df['Размер.компании'] = df['Размер.компании'].replace(['свыше 1000 человек'], 5)
    # print(df['Размер.компании'].unique())

    # Prepare Тип компании number value
    df['Тип.компании'] = df['Тип.компании'].replace(['Другая'], 0)
    df['Тип.компании'] = df['Тип.компании'].replace(['Продуктовая'], 1)
    df['Тип.компании'] = df['Тип.компании'].replace(['Аутсорсинговая'], 2)
    df['Тип.компании'] = df['Тип.компании'].replace(['Стартап'], 3)
    df['Тип.компании'] = df['Тип.компании'].replace(['Аутстаффинговая'], 4)
    # print(df['Тип.компании'].unique())

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from IPython.core.pylabtools import figsize
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')
pyplot.rcParams['font.size'] = 22
sns.set(font_scale=2)


# Обучение и тестирование модели
def fit_and_evaluate(model):
    model.fit(X, y)
    model_pred = model.predict(X_test)
    return model_pred


# Таблица отсутствующих значений в процентном соотношении
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Отсутствующие значения', 1: '% от всех значений'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% от всех значений', ascending=False).round(1)
    print("В data frame содержится " + str(df.shape[1]) + " столбцов.\n"
                                                          "Всего " + str(
        mis_val_table_ren_columns.shape[0]) +
          " столбец с отсутствующими значениями.")
    return mis_val_table_ren_columns


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = pyplot.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Объем обучающей выборки")
    axes.set_ylabel("-MAE")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring='neg_mean_absolute_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="-MAE обучения")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="-MAE кросс-валидации")
    axes.legend(loc="best")


path_less = "D:\\Лабы по машинному обучению\\Лаба4\\13\\winequality-red.csv"

data_less = pd.read_csv(path_less, delimiter=";")

# Первичный анализ признаков

# Мета данные признаков
print(f'Мета данные признаков{data_less.info()}')

# Процент пропущенных значений для каждого признака
missing_df = missing_values_table(data_less)
print(missing_df, '\n')

missing_columns = list(missing_df[missing_df['% от всех значений'] >= 50].index)
data = data_less.drop(columns=list(missing_columns))
print(f'Столбцы, которые нужно убрать из выборки: {missing_columns}\n')

# Замена Not Available на not a number
data = data.replace({'Not Available': np.nan})

print(f'Первичный анализ: \n{data.describe()}\n')

print(f'Медиана: \n{data.median()}\nМода: \n{data.mode()}')

# Первичный визуальный анализ признаков

# Гиcтограмма для параметра quality
data['quality'].hist()
pyplot.xlabel('Качество', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра quality', size=28)
pyplot.show()

# Гиcтограмма для параметра fixed acidity
data['fixed acidity'].hist(bins=20)
pyplot.xlabel('Фиксированная кислотность', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра fixed acidity', size=28)
pyplot.show()

# Гиcтограмма для параметра volatile acidity
data['volatile acidity'].hist(bins=20)
pyplot.xlabel('Летучая кислотность', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра volatile acidity', size=28)
pyplot.show()

# Гиcтограмма для параметра citric acid
data['citric acid'].hist(bins=20)
pyplot.xlabel('Лимонная кислота', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра citric acid', size=28)
pyplot.show()

# Гиcтограмма для параметра residual sugar
data['residual sugar'].hist(bins=20)
pyplot.xlabel('Остаточный сахар', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра residual sugar', size=28)
pyplot.show()

# Гиcтограмма для параметра chlorides
data['chlorides'].hist(bins=20)
pyplot.xlabel('Хлориды', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра chlorides', size=28)
pyplot.show()

# Гиcтограмма для параметра free sulfur dioxide
data['free sulfur dioxide'].hist(bins=20)
pyplot.xlabel('Свободный диоксид серы', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра free sulfur dioxide', size=28)
pyplot.show()

# Гиcтограмма для параметра total sulfur dioxide
data['total sulfur dioxide'].hist(bins=20)
pyplot.xlabel('Общий диоксид серы', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра total sulfur dioxide', size=28)
pyplot.show()

# Гиcтограмма для параметра density
data['density'].hist(bins=20)
pyplot.xlabel('Плотность', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра density', size=28)
pyplot.show()

# Гиcтограмма для параметра pH
data['pH'].hist(bins=20)
pyplot.xlabel('pH', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра pH', size=28)
pyplot.show()

# Гиcтограмма для параметра sulphates
data['sulphates'].hist(bins=20)
pyplot.xlabel('Сульфаты', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра sulphates', size=28)
pyplot.show()

# Гиcтограмма для параметра alcohol
data['alcohol'].hist(bins=20)
pyplot.xlabel('Алкоголь', size=20)
pyplot.ylabel('Количество наблюдений', size=20)
pyplot.title('Гитограмма для параметра alcohol', size=28)
pyplot.show()

# Зависимость стоимости от количества комнат
pyplot.scatter(data['alcohol'], data['volatile acidity'])
pyplot.xlabel('Алкоголь', size=20)
pyplot.ylabel('Летучая кислотность', size=20)
pyplot.show()

# Диаграмма размаха цены здания в зависимости от типа
data.boxplot(column=['volatile acidity'])
pyplot.xlabel('Летучая кислотность', size=20)
pyplot.show()

# Преобразование данных


# Закономерности, особенности данных

# Все корреляции с целевым параметром
correlations_data = data.corr()['quality'].sort_values()
print(f'Корреляции с целевым параметром: \n{correlations_data}')

features = data.copy()

# Разделение на обучающую и тестовую выборки
quality = features[features['quality'].notnull()]

features = quality.drop(columns='quality')
targets = pd.DataFrame(quality['quality'])

X, X_test, y, y_test = train_test_split(features, targets, test_size=0.15, random_state=42)

print(f'Количество наблюдений в обучающей выборке: {X.shape[0]}\n'
      f'Количество наблюдений в тестовой выборке: {X_test.shape[0]}\n')

# Масштабирование значений
scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(X)

X = scaler.transform(X)
X_test = scaler.transform(X_test)

y = np.array(y).reshape((-1,))
y_test = np.array(y_test).reshape((-1,))

print('Количество отсутствующих значений в наборе для обучения: ', np.sum(np.isnan(X)))
print('Количество отсутствующих значений в наборе для тестирования:  ', np.sum(np.isnan(X_test)))

# Построение кривой обучения
title = "Кривая обучения"
estimator = RandomForestRegressor(n_estimators=150)
plot_learning_curve(estimator, title, X, y)
pyplot.show()

# Построение кривой валидации
param_range = [100, 150, 200, 250, 300]
train_scores, test_scores = validation_curve(KNeighborsRegressor(), X, y,
                                             param_name="n_neighbors", param_range=param_range,
                                             scoring='neg_mean_absolute_error')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

pyplot.title("Кривая валидации")
pyplot.xlabel("n_neighbors")
pyplot.ylabel("-MAE")
lw = 2
pyplot.semilogx(param_range, train_scores_mean, label="-MAE обучения",
             color="darkorange", lw=lw)
pyplot.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
pyplot.semilogx(param_range, test_scores_mean, label="-MAE кросс-валидации",
             color="navy", lw=lw)
pyplot.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
pyplot.legend(loc="best")
pyplot.show()

# Модель
rfr = RandomForestRegressor()
rfr_pred_array = fit_and_evaluate(rfr)

print(f'R Squared для модели KNeighborsRegressor = {r2_score(y_test, rfr_pred_array)}')

# Прогноз
pred = rfr.predict(X_test)
print(f'Прогноз: \n{pred}\nЗначения: \n{y_test}')
sns.kdeplot(pred, label='Прогноз')
sns.kdeplot(y_test, label='Значения')

pyplot.xlabel('Качество')
pyplot.ylabel('Плотность')
pyplot.title('Значения и прогноз')
pyplot.legend(loc="best")
pyplot.show()

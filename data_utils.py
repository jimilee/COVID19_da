import math_utils
import pandas as pd
import numpy as np
import pickle
import itertools
import gc
import math
import matplotlib.pyplot as plt
import dateutil.easter as easter
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


covid_data_origin = pd.read_excel('D:/_workspace/covid_da/코로나바이러스감염증-19_확진환자_발생현황_220111.xlsx', skiprows=range(4))
covid_data_origin = covid_data_origin.drop(0)
covid_data_origin.columns=['date','total','domestic','inflow','death']
covid_data_origin.date=pd.to_datetime(covid_data_origin['date'])
for col in ['total','domestic','inflow','death']:
    covid_data_origin[col] = covid_data_origin[col].replace(['-'],'0').astype(np.float64)
print(covid_data_origin)

def fit_model(X_tr, X_va=None):
    """Scale the data, fit a model, plot the training history and validate the model"""
    score_list = []
    start_time = datetime.now()
    oof = pd.Series(0.0, index=train_df.index)
    # Preprocess the data
    X_tr_f = X_tr[features]
    preproc = StandardScaler()
    X_tr_f = preproc.fit_transform(X_tr_f)
    y_tr = X_tr.total.values.reshape(-1, 1)

    # Train the model
    #model = LinearRegression()
    #model = HuberRegressor(epsilon=1.20, max_iter=500)
    #model = Ridge()
    model = RandomForestRegressor()
    model.fit(X_tr_f, np.log1p(y_tr).ravel())

    if X_va is not None:
        # Preprocess the validation data
        X_va_f = X_va[features]
        X_va_f = preproc.transform(X_va_f)
        y_va = X_va.total.values.reshape(-1, 1)

        # Inference for validation
        y_va_pred = np.exp(model.predict(X_va_f)).reshape(-1, 1)
        oof.update(pd.Series(y_va_pred.ravel(), index=X_va.index))

        # Evaluation: Execution time and SMAPE
        # y_va_pred *= LOSS_CORRECTION
        smape = np.mean(math_utils.smape_loss(y_va, y_va_pred))

        score_list.append(smape)

        # Plot y_true vs. y_pred
        plt.figure(figsize=(5, 5))
        plt.scatter(y_va, y_va_pred, s=1, color='r')
        # plt.scatter(np.log(y_va), np.log(y_va_pred), s=1, color='g')
        plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], '--', color='k')
        plt.gca().set_aspect('equal')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title('OOF Predictions')
        plt.show()

    return preproc, model

def plot_three_years_combination(engineer):
    demo_df = pd.DataFrame({'date':pd.date_range('2020-01-20', '2023-01-20', freq='D'),
                        })
    demo_df.set_index('date', inplace=True, drop=False)
    demo_df['total'] = covid_data_origin.total
    demo_df['total'] = demo_df['total'].replace(np.nan, 0)
    demo_df = engineer(demo_df)
    print(demo_df[features].head())
    demo_df['total'] = np.exp(model.predict(preproc.transform(demo_df[features])))
    plt.figure(figsize=(20, 6))
    # plt.plot(np.arange(len(demo_df)), demo_df.total, label='prediction')
    train_subset = train_df
    # plt.scatter(np.arange(len(train_subset)), train_subset.total, label='true', alpha=0.5, color='red', s=3)
    plt.scatter(np.arange(len(train_subset)), covid_data_origin.death, label='death', alpha=0.5, color='black', s=3)
    plt.scatter(np.arange(len(train_subset)), (covid_data_origin.death/covid_data_origin.total)*100, label='death per total(%)', alpha=0.5, color='red', s=3)
    plt.legend()
    plt.title('Predictions and true total for 3 years')
    plt.show()


train_df = math_utils.engineer(covid_data_origin)
print(train_df.head())
train_df['date'] = covid_data_origin.date
train_df['total'] = covid_data_origin.total.astype(np.float64)

covid_data_origin_test = covid_data_origin.sample(frac=0.4,random_state=60)
print(covid_data_origin_test.head())
test_df = math_utils.engineer(covid_data_origin_test)

features = test_df.columns

for df in [train_df, test_df]:
    df[features] = df[features].astype(np.float64)

print(list(features))
print(train_df.head())
train_df = train_df.dropna()
#
# preproc, model = fit_model(train_df)
#
# plot_three_years_combination(math_utils.engineer)
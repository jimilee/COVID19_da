import pandas as pd
import numpy as np
from pycaret.time_series import *
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from pycaret.datasets import get_data

# Convert dataframe for pycaret.
covid_data_origin = pd.read_excel('D:/_workspace/covid_da/코로나바이러스감염증-19_확진환자_발생현황_220111.xlsx', skiprows=range(4))
covid_data_origin = covid_data_origin.drop(0)
covid_data_origin.columns=['Date','Total','Domestic','Inflow','Death']
idx = pd.to_datetime(covid_data_origin['Date'])

total_list = covid_data_origin['Total'].replace(['-'],'0').astype(np.float64).tolist()
covid_data = pd.Series(total_list, idx)

print(covid_data)
print(type(covid_data.index))

# with functional API
setup(covid_data, fh = 7, fold = 3, session_id = 123)

# with new object-oriented API
exp = TimeSeriesExperiment()
exp.setup(covid_data, fh = 7, fold = 3, session_id = 123)

check_stats()


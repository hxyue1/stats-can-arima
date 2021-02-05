import streamlit as st
import stats_can
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from datetime import timedelta, datetime

data_vectors_dict = {'Monthly GDP Growth Rate (Annualised)': 'v65201210', 
                     'Monthly Inflation Rate (Annualised)': 'v41690973', 
                     'Unemployment Rate':'v2062815'
                     }

def get_data(data_source):
    df = stats_can.sc.vectors_to_df(data_vectors_dict[data_source], periods = 300)
    
    if data_source != 'Unemployment Rate':
        out_df = (df-df.shift())/df*100*12
        out_df = out_df.iloc[1:]
    else:
        out_df = df
    
    out_df.columns = [data_source]
    return(out_df)
    
st.header('Data')
data_source = st.selectbox(label='Data source', options=list(data_vectors_dict.keys()))
data = get_data(data_source)
st.dataframe(data)
st.subheader('Time plot of the data')
st.line_chart(data)

st.header('Checking for time dependcies')
st.markdown('Here we plot the ACF and PACF of the time series to identify patterns of dependency which can be potentially exploited in our modelling. Note that lag zero has been omitted.')

fig, ax = plt.subplots(2,1)
plot_acf(data, zero=False, ax=ax[0])
plot_pacf(data, zero=False, ax=ax[1])
#Needed to adjust spacing between plots
fig.tight_layout(pad=1.5)
st.pyplot(fig)

###Modelling
st.header('ARIMA Modelling')
AR = st.slider('Number of Autoregressive terms', min_value=0, max_value=10, value=1)
MA = st.slider('Number of Moving Average terms', min_value=0, max_value=10)
I = st.slider('Order of Integration', min_value=0, max_value=3)

model = ARIMA(data, order=(AR, I, MA))
res = model.fit()
st.markdown(res.summary().as_html(), unsafe_allow_html=True)

st.header('Residual plots')
st.markdown('We plot the ACF and PACF of the residuals to check if there are any dependencies that the model has not accounted for. If the model is well specified, the ACF and PACF values should be within the shaded light blue area. If not, try playing around with different ARIMA specifications.')

fig, ax = plt.subplots(2,1)
plot_acf(res.resid, zero=False, ax=ax[0])
plot_pacf(res.resid, zero=False, ax=ax[1])
fig.tight_layout(pad=1.5)
st.pyplot(fig)

###Forecasting
st.header('Forecasting Evaluation')
start_date = st.slider('Forecasting start date', min_value=data.index[150].date(), max_value=data.index.max().date(), value=data.index[215].date(),  step=timedelta(days=30))
end_date = st.slider('Forecasting end date', min_value=start_date, max_value=datetime(2020, 1, 1, 0, 0).date(), value=datetime(2019,6,1,0,0).date(),  step=timedelta(days=30))

dates = data.index.to_pydatetime()
get_date = lambda x: x.date()
date_transformer = np.vectorize(get_date)
transformed_dates = date_transformer(dates)
start_idx = np.where(transformed_dates<=start_date)[0].max()
end_idx = np.where(transformed_dates<=end_date)[0].max()

forecasts = []
for period in range(start_idx, end_idx+1):
    data_temp=data.iloc[:period]
    model_temp = ARIMA(data_temp, order=(AR, MA, I))
    res_temp = model_temp.fit()
    forecasts.append(res_temp.forecast()[0])

###Plotting forecasts
forecasts_df = data.iloc[start_idx:end_idx+1].copy()
forecasts_df.loc[:,'Forecasted'] = forecasts
forecasts_df.columns = ['Actual', 'Forecasted']

st.subheader('Actual vs Forecasted')
st.dataframe(forecasts_df)
st.line_chart(forecasts_df)

st.subheader('Evaluation Metrics')
RMSE = np.mean((forecasts_df['Actual'] - forecasts_df['Forecasted'])**2)**0.5
MAE = np.mean(abs(forecasts_df['Actual'] - forecasts_df['Forecasted']))
MAPE = np.mean(abs((forecasts_df['Actual'] - forecasts_df['Forecasted'])/forecasts_df['Actual']))*100

st.text('RMSE is ' + str(np.round(RMSE, 3)))
st.text('MAE is ' + str(np.round(MAE, 3)))
st.text('MAPE is ' + str(np.round(MAPE, 3)))




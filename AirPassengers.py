# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:26:48 2023

@author: domingosdeeulariadumba
"""


# IMPORTING REQUIRED LIBRARIES

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from scipy import stats
from scipy.stats import linregress
from scipy.signal import periodogram
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')




# EXPLORATORY DATA ANALYSIS

    '''
    Below, we import the dataset, we check the info to ensure the data type
    and possible missing values.
    We then converted the 'Month' column to datetime to use it as the index
    of the dataframe.
    And lastly we plot the time series.
    '''
df = pd.read_csv("AirPassengers.csv")

df.info()

df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

df.index = df['Month']

del df['Month']
    
    # One-sided plot    
plt.suptitle('Number US Airline Passengers from 1949 to 1960', 
             y = 0.9, fontsize = 18)
sb.lineplot(data=df, legend = False)
plt.ylabel('Number of Passengers', fontsize = 16)


x = df.index.values
y = df['#Passengers'].values

    # Two-sided plot
plt.plot(figsize=(16,10), dpi= 240)
plt.fill_between(x, y1=y, y2=-y, alpha=0.5, linewidth=2, color='grey')
plt.ylim(-800, 800)
plt.title('Air Passengers (Two-Side View)', fontsize=16)
plt.hlines(y=0, xmin=np.min(df.index), xmax=np.max(df.index), linewidth=4,
           linestyles='-.')
plt.show()


    ''' CHECKING STATIONARITY '''

    ''' In order to simplify the forecasting process, the time serie needs to be stationary. 
    So, although we could take a conclusion about this detail from the prior plots, in some 
    cases it may not  be that clear. That is, we need to check wether our data is non-stationary
    or stationary to stress the visual analysis. It is done using statistical tests such as
    Augmented Dickey-Fuller (ADF) Test and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test. 
    We'll implement the ADF method.'''

def check_stationarity(df):
    
    adft = adf(df,autolag="AIC")
    
    scores_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3],
                                    adft[4]['1%'], adft[4]['5%'],
                                    adft[4]['10%']]  ,
                          "Metric":["Test Statistics",
                                    "p-value","No. of lags used",
                                    "Number of observations used", 
                                    "critical value (1%)",
                                    "critical value (5%)",
                                    "critical value (10%)"]})
     
    if scores_df['Values'][1]  > 0.05 and scores_df['Values'][0]>scores_df['Values'][5]:
        print('This serie is not stationary!!!')
        
    return

check_stationarity(df)

   
     ''' ANALYSIS OF COMPONENTS'''
    
    # What trend do we have?

        '''
        We apply the moving average for a window of 6 and 12 months to find
        out the trend of this serie (although by looking at the first plot
        it becomes already clear)
        
        '''
MA_6 = df.rolling(6).mean()
MA_12 = df.rolling(12).mean()

plt.plot(df, color="blue",label="Original Data")
plt.plot(MA_6, color="red", label="Rolling Mean for 6 Months")
plt.plot(MA_12, color="black",
         label = "Rolling Mean for 12 Months")
plt.legend(fontsize = 18)


    # What about seasonality?

def custom_periodogram(ts, detrend='linear', ax=None,
                       freq_range=(1/12, 1), color='b'):
    
        # Determine the number of data points
        N = len(ts)

        # Compute the periodogram
        frequencies, spectrum = periodogram(ts, detrend=detrend,
                                            scaling='spectrum')

        # Create a plot if ax is not provided
        if ax is None:
            _, ax = plt.subplots()

        # Filter frequencies within the specified range
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        filtered_freq = frequencies[mask]
        spectrum = spectrum[mask]

        # Define additional frequencies and labels for annual, biennial...
        #... triennial, quadrennial, sexennial, decennial.
        additional_freqs = [1, 1/2, 1/3, 1/4, 1/6, 2, 4, 6, 12, 1/8, 1/10]
        additional_labels = [
            "Annual (1 year)", "Biennial (2 years)",
            "Triennial (3 years)", "Quadrennial (4 years)",
            "Sexennial (6 years)", "Semiannual (0.5 year)",
            "Quarterly (0.25 year)", "Bimonthly (1/6 year)",
            "Monthly (1/12 year)", "Octennial (8 years)",
            "Decennial (10 years)"]

        # Plot the periodogram for the specified frequencies
        ax.step(filtered_freq, spectrum, color=color)
        ax.set_xscale("log")
        ax.set_xticks(additional_freqs)
        ax.set_xticklabels(additional_labels, rotation=35, fontsize=16)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Variance", fontsize=12)
        ax.set_title(f"Custom Periodogram (N={N})", fontsize=18)

        return ax

custom_periodogram(df['#Passengers'])

    '''
    As we only have montly observations, the periodogram does not inform much
    about annual seasonality considering this temporal dimension. Thus, we then
    apply the seasonal plot, which is recommended for few observations.
    '''

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sb.color_palette("rocket", n_colors=X[period].nunique())
    ax = sb.lineplot(
        x=freq, y=y, hue=period, data=X, ax=ax, palette=palette)
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    plt.legend(fontsize = 18, loc = 'best')
    
    return ax

X_season = df.copy()

X_season['date'] = df.index.values

X_season['year'] = X_season['date'].dt.year

X_season['month'] = X_season['date'].dt.month

y = X_season['#Passengers']

fig, ax = plt.subplots(figsize=(11, 6))
seasonal_plot(X_season, y, period = 'year', freq = 'month')

    ''' 
    From the seasonal plot, it is noticed an increase from one year to 
    another in the number of passengers. Regarding seasonality, it is also
    possible to observe a spike from june to august (this marks the summer time,
    and this spike is understandable as there is more tourists visiting the
    country, mostly due to this season being the longest educational break
    of the year)
    '''

    # What about serial Dependency?
    
    '''
    We then analyse the features that does not depend on time, plotting the lagged 
    version of the time series against his previous version. To simplify this
    step, we defined two functions: the first one to illustrate the lag plots,
    and the second to create a correlogram, in order to visualize the lagged
    version of the series that carries relevant information.    
    '''
    
def plot_lags(df, n_lags = 10):
    
    fig, axes = plt.subplots(2, n_lags//2, figsize=(14, 10), sharex=False, sharey=True,
                             dpi=240)
    
    for i, ax in enumerate(axes.flatten()[:n_lags]):
        lag_data = pd.DataFrame({'x': df['#Passengers'],
                                 'y': df['#Passengers'].shift(i+1)}).dropna()
        
        x, y = lag_data['x'], lag_data['y']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_line = [(slope * xi) + intercept for xi in x]   
        ax.scatter(x, y, c = 'k', alpha = 0.6)
        ax.plot(x, regression_line, color='m', label=f'{r_value**2:.2f}')
        ax.set_title(f'Lag {i+1}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.suptitle('Lag Plots for Passengers', fontsize = 16, fontweight = 'bold', y = 1.05)
    plt.show()
    
    return

    ''' Plotting the lagged versions'''
plot_lags(df,8)

    ''' Creating the function to generate correlogram'''
def plot_correlogram(df, n_lags = 10):
    
    plot_pacf(df['#Passengers'], color = 'b', lags = n_lags)
    plt.title("Partial Autocorrelation Frequency Plot", fontsize=16,
              fontweight='bold', y = 1.05)
    plt.show()

    return 

plot_correlogram(df)


    # DECOMPOSITION
    
    '''
    Next, we illustrate the time serie, its trend and seasonal patterns, and
    the residual (error) considering a multiplicative and additive 
    decomposition.
    '''

add_dec = seasonal_decompose(df, model='additive')
mtp_dec = seasonal_decompose(df, model='multiplicative')

add_dec.plot().suptitle('Additive Decomposition', y = 1.05, fontsize = 16)
mtp_dec.plot().suptitle('Multiplicative Decomposition',
                        y = 1.05, fontsize = 16)
plt.show()

    '''
    The additive decomposition looks more suitable for this case, since it
    tends to capture more information out of two main patterns (trend and 
                                                                season)
    '''


    # FORECASTING
    
    '''
    We will finally make forecasts, using SARIMA, since we have a season in
    this time series.
    Firstly, we create a function to split the dataset. the result is a tuple
    of train and test set. We then plot the serie considering this division.    
    '''

def split_data(df, train_proportion = 0.8):
    
    train_size = round(len(df)*train_proportion)
    train_set = df.iloc[:train_size].rename(columns= {'#Passengers':
                                                      'train_set'})
    test_set = df.iloc[train_size:].rename(columns= {'#Passengers':
                                                      'test_set'})
    
    return train_set, test_set


train_data, test_data = split_data(df)[0], split_data(df)[1]

plt.suptitle("Train/Test split for Passenger Data", y = 0.9, fontsize = 20)
plt.plot(train_data, color = "b", label = 'Train Set')
plt.plot(test_data, color = "g", label = 'Test Set', linestyle = '--')
plt.ylabel("Passenger Per Year")
plt.legend(fontsize = 18)
plt.show()

    '''
    The auto_select variable makes an automatic grid search for the optimal
    parameters of the two patterns of the serie, trend and season (it is the
    ideal number of autoregressigve terms/lagged values, the number of 
    differences required to make the serie stationary, and the number of 
    moving average), considering 12 as the number of time steps in one seasonal
    cycle, since we have monthly data and the seasonality is yearly.
    '''
    
    '''
    grid searching the optimal combination
    '''
auto_select = auto_arima(train_data, seasonal = True, m = 12,
                   stepwise = True, trace = True)

     '''
     Training and fitting the model
     '''
model = SARIMAX(train_data, order=auto_select.order,
                       seasonal_order = auto_select.seasonal_order).fit()

     '''
     Making forecast and checking RMSE
     '''
steps_forecast = model.get_forecast(steps = len(test_data))

forecasted_values = steps_forecast.predicted_mean

print('RMSE:',
      mean_squared_error(test_data, forecasted_values, 
                         squared = False))

     '''
     Plotting the Results
     '''
plt.plot(train_data, color = 'b', label="Train Data")
plt.plot(test_data, color = 'r', label="Test Data")
plt.plot(forecasted_values, color = 'g', label="Forecasted Values",
         linestyle = '--')
plt.title("Number of Passengers Forecast", fontsize = 16)
plt.legend(fontsize = 14)
plt.show()

    '''
    Forecasting for 1961 (12 months ahead), it is until
    december of this year.
    '''
steps_12 = model.get_forecast(steps = len(test_data)+12)
forecast_12 = steps_12.predicted_mean


    '''
    In order to forecast the Number of Passengers ahead of 1960, for this case
    until 1961, we decide to compute the Confidence Intervals (95% for its 
    default value), so we can better visualize the forecasting accuracy.
    '''
c_i = steps_12.conf_int()
lower_bound = c_i['lower train_set']
upper_bound = c_i['upper train_set']
period = c_i.index.values

    '''
    Displaying the results with confidence interval.
    '''
    
plt.plot(figsize=(16,10), dpi= 240)
plt.fill_between(period, y1=upper_bound, y2=lower_bound,
                 alpha=0.5, linewidth=2, color='grey')
plt.plot(train_data, color = 'b', label="Train Data")
plt.plot(test_data, color = 'r', label="Test Data")
plt.plot(forecast_12, color = 'g',
         label="Forecasted Values for One Year",
         linestyle = '--')
plt.title("Number of Passengers Forecast in US for 1961",
          fontsize = 16, bold = True)
plt.legend(fontsize = 14)
plt.show()
-----------------------------------END-----------------------------------------

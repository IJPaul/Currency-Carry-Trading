
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[245]:


import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
import math
from datetime import datetime
from datetime import timedelta
import random


# In[4]:





# In[7]:


"""
Converts a string t representing a date into a pandas Timestamp object

Parameter: t is the date
Precondition: t is a date in the form of YYYY-MM-DD
"""
def toTimestamp(t):
    return pd.Timestamp(t + ' 0:00:00')


# In[8]:


"""
Returns the rows of the dataframe that correspond
to times between range of start_date and end_date inclusive

Parameter df: df is a dataframe
Precondition: df is a pandas dataframe object

Paremeters start_date, end_date:
Preconditons: start_date and end_date are both date strings of the form YYYY-MM-DD
"""
def dataframeBetween(df, start_date, end_date):
    t1 = toTimestamp(start_date)
    t2 = toTimestamp(end_date)
    return df.loc[(df['Dates'] >= t1) & (df['Dates'] <= t2)]


# In[9]:


def indexOfDate(df, date):
    t = toTimestamp(date)
    return df.loc[df['Dates'] == t].index[0]

def dateOfIndex(df, index):
    return df['Dates'][index]


# In[210]:


#audjpy_df = pd.read_excel('/Users/Ian/Downloads/AUDJPY_Data.xlsx')
audjpy_df = pd.read_excel('/Users/Ian/Downloads/BacktestDataAUDJPY.xlsx')
col = audjpy_df['PX_LAST']
dataframeBetween(audjpy_df, '2015-10-01', '2015-12-06')


# In[11]:


"""
Returns an array of the N day simple moving average (SMA) 
of df[col] for each day between start date and end date inclusive

Parameter df: df is a dataframe
Precondition: df is pandas dataframe object

Parameter col: col is the column to caculate SMA of
Preconditon: col is a string

Paremeters start_date, end_date:
Preconditons: start_date and end_date are both date strings of the form YYYY-MM-DD
"""
def SMA(df, col, n_day, start_date, end_date):
    sma_arr = []
    sdf = dataframeBetween(df, start_date, end_date)
    sum_ = 0
    for i in range(0, len(sdf[col]) - (n_day-1)):
        n_day_arr = sdf[col][i : i + n_day]
        sum_ = 0
        for price in n_day_arr:
            sum_ = price + sum_
        sma_arr.append(sum_ / n_day)
    return sma_arr


# In[12]:


"""
Returns the weighting multiplier used
in an N day EMA calculation

Parameter n_day: The length of the MA period
Precondition: n_day is an int
"""
def weightingMult(n_day):
    return 2.0 / (n_day + 1.0)


# In[220]:


"""
Returns an array of the N day exponential moving average (SMA) 
of df[col] for each day between start date and end date inclusive.
EMA is calculated as follows:
EMA_0 = N day SMA
EMA_t = (Last Price - EMA_t-1 ] x multiplier) + EMA_t-1

Parameter df: df is a dataframe
Precondition: df is pandas dataframe object

Parameter col: col is the column to caculate EMA of
Preconditon: col is a string

Paremeters start_date, end_date:
Preconditons: start_date and end_date are both date strings of the form YYYY-MM-DD
"""
def EMA(df, col, n_day, start_date, end_date):
    ema_arr = []
    sdf = dataframeBetween(df, start_date, end_date)
    sum_ = 0
    weighting_mult = weightingMult(n_day)

    # SMA of first n days initalizes EMA
    n_day_arr = sdf[col].iloc[:n_day]
    initial_ema = sum(n_day_arr) / n_day
    ema_arr.append(initial_ema)
    #print(sdf[col])
    #print(len(sdf[col].iloc[n_day : len(sdf[col])]))
    for price in sdf[col].iloc[n_day : len(sdf[col])]:    
        ema_arr.append((price - ema_arr[-1]) * weighting_mult + ema_arr[-1])
   
    return ema_arr

EMA(audjpy_df, 'PX_LAST', 20, '2015-12-04', '2016-12-06')


# In[185]:


audjpy_df = pd.read_excel('/Users/Ian/Downloads/AUDJPY_Data.xlsx')
#col = audjpy_dfhttp://localhost:8888/notebooks/Currency%20Trading.ipynb#['PX_LAST']
#print(EMA(audjpy_df, 'PX_LAST', 90, "2018-04-07", "2018-11-01"))

def movingAverageRibbons(df, col, n_day, start_date, end_date):
    moving_avgs = {'5-day' : None, '10-day' : None, '20-day' : None,
               '30-day' : None,'40-day' : None, '50-day' : None,
               '75-day' : None, '100-day' : None, '200-day' : None} 
    for key in moving_avgs.keys():
        days_in_avg = int(key.split('-')[0])

        moving_avgs[key] = EMA(df, col, days_in_avg , start_date, end_date)
    print(moving_avgs)
print(EMA(audjpy_df, 'PX_LAST', 20, '2015-12-04', '2016-12-06'))
#movingAverageRibbons(audjpy_df, 'PX_LAST', 20, '2015-10-04', '2015-12-06')


# In[186]:



def plotPrice(df, start_date , end_date, xlab = "Date", ylab = "Price"):
    import matplotlib.dates as mdates
    
    
    sub_df = dataframeBetween(df, start_date, end_date)
    dates = sub_df['Dates']
    price = sub_df['PX_LAST']
    
    fig, ax = plt.subplots()
    
    #myFmt = mdates.DateFormatter('%b %d %y')
    #ax.xaxis.set_major_formatter(myFmt)
    
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    title = ylab + " between " + start_date + " and " + end_date
    fig.suptitle(title, fontsize=14)
 
    return plt.plot(dates, price)
    plt.show()
    
plotPrice(audjpy_df, "2015-10-15", "2018-11-15")


# In[197]:


"""
Returns a list of the peaks in df[col] between
start date and end_date in df[col]. Smaller W 
values correspond to the width of peaks. Local maxima
will have a smaller W and global maxima will have larger W

Parameter w: w is the list of peak widths to search for
Precondition: w is a 1-D array of numbers
"""
def findPeaks(df, col, start_date, end_date, w=np.linspace(1,10)):
    sdf = dataframeBetween(df, start_date, end_date)
    column = sdf[col].values
    return column[signal.find_peaks_cwt(column, w)]


# In[219]:


"""
Returns a list of the troughs in df[col] between
start_date and end_date in df[col]. Smaller W 
values correspond to the width of peaks. Local maxima
will have a smaller W and global maxima will have larger W

Parameter w: w is the list of peak widths to search for
Precondition: w is a 1-D array of numbers
"""
def findTroughs(df, col, start_date, end_date, w=np.linspace(1,10)):
    sdf = dataframeBetween(df, start_date, end_date)
    column = (-1 * sdf[col]).values
    #print(column)
    #print(signal.find_peaks_cwt(column, w))
    return -1* np.array(column[signal.find_peaks_cwt(column, w)])

findTroughs(audjpy_df, 'PX_LAST', '2014-12-02', '2015-05-20')


# In[231]:


data = audjpy_df['PX_LAST']
#print(findPeaks(audjpy_df, 'PX_LAST', '2016-12-01', '2018-12-21', np.arange(1,100)))
#print(findTroughs(audjpy_df, 'PX_LAST', '2016-12-01', '2016-12-21', [4,10]))
def plotPeaksTroughs(df, col, start_date, end_date, w=np.linspace(1,10), pc = 'g', tc = 'r'):
    plotPrice(df, start_date, end_date)
    for peak in findPeaks(df, col, start_date, end_date, w):
        plt.axhline(y=peak , color=pc, linestyle='-')
    for trough in findTroughs(df, col, start_date, end_date, w):
        plt.axhline(y=trough , color=tc, linestyle='-')

plotPeaksTroughs(audjpy_df, 'PX_LAST', '2014-12-01', '2015-12-21')


# In[24]:


"""
Returns True if the current price of the security is at a retracement level.
Otherwise, False.
"""
def isRetracementLevel(df, col, price, start_date, end_date, w):
    rls = [0.0, 0.236, 0.382, 0.50, 0.618, 1.0]
    is_rl = False
    peaks = findPeaks(df, col, start_date, end_date, w)
    troughs = findTroughs(df, col, start_date, end_date, w)
    if peaks is None: 
        peaks = []
    if troughs is None:
        troughs = []
    peak = max(peaks)
    trough = min(troughs)
    for rl in rls:
        if abs(price - (trough + (rl * (peak - trough)))) < 0.01:
            is_rl = True
    return is_rl


# In[229]:


"""
Plots the Fibonacci retracement level prices as points on a time series plot
"""
def plotRetracementLevels(df, col, start_date, end_date, w = np.linspace(1,10), c = 'r'):
    plotPrice(df, start_date, end_date)
    sub_df = dataframeBetween(df, start_date, end_date)
    column = sub_df[col]
    
    for i in range(len(column)):
        price = column.iloc[i]
        if isRetracementLevel(sub_df, col, price, start_date, end_date, w):
            
            plt.plot(sub_df['Dates'].iloc[i], price, marker='o', markersize=6, color=c)
            
column = dataframeBetween(audjpy_df, '2016-12-01', '2016-12-21')['PX_LAST']
plotRetracementLevels(audjpy_df, 'PX_LAST', '2018-01-01', '2018-06-01')


# In[213]:


"""
Returns the N day standard deviation of df[col] between
start_date and end_date
"""
def ndayStdDev(df, col, n_day, start_date, end_date):
    sd_arr = []
    sdf = dataframeBetween(df, start_date, end_date)
    sma = SMA(df, col, n_day, start_date , end_date)
    
    for i in range(n_day, len(sdf[col])):
        sum_of_squares = 0
        n_day_arr = sdf[col].iloc[i-n_day:i]
        sma_i = sma[i-n_day]
    
        for price in n_day_arr:
            sum_of_squares = sum_of_squares + (price - sma_i)**2
        
        sd = (sum_of_squares / (n_day  - 1))**0.5
        sd_arr.append(sd)
    return sd_arr


# In[236]:


"""
Returns the upper bollinger band array of prices where each element is the N period EMA + Kσ up to and including
price. σ is the N period standard deviation in price.

Parameter n: the N-period to observe
Precondition: n is an int

Parameter k: the number of std deviations from the MA
Preconditon: k is a double

Parameter: price
Precondition: price is the current price of the currency pair
"""
def upperBollinger(df, col, n_day, k, start_date, end_date):
    sma = SMA(df, col, n_day, start_date, end_date)
    sdf = dataframeBetween(df, start_date, end_date)
    ub = []
    for i in range(len(sdf[col]) - n_day):
        ub.append(sma[i] + k*((ndayStdDev(df, col, n_day, start_date, end_date))[i]) ) 
    return ub


# In[237]:


"""
Returns the lower bollinger band array of prices where each element is the N period EMA - Kσ up to and including
price. σ is the N period standard deviation in price.
"""
def lowerBollinger(df, col, n_day, k, start_date, end_date):
    sma = SMA(df, col, n_day, start_date, end_date)
    sdf = dataframeBetween(df, start_date, end_date)
    lb = []
    for i in range(len(sdf[col]) - n_day):
        lb.append(sma[i] - k*((ndayStdDev(df, col, n_day, start_date, end_date))[i]))
    return lb


# In[100]:


"""
Converts a Timestamp object to a date string of the form YYYY-MM-DD
"""
def timeStampToStr(t):
    year = str(t.year)
    month = str(t.month)
    day = str(t.day)
    return year + '-' + month + '-' + day


# In[238]:


"""
Plots Bollinger Bands of df[col] between start_date and end_date

Parameter k: the multipler of standard deviation for the Bollinger bands
Precondition: k is a positive int
"""
def bollingerBands(df, col, n_day, k, start_date, end_date):
    sub_df = dataframeBetween(df, start_date, end_date)
    dates = sub_df['Dates']
    lower_bollinger = lowerBollinger(df, col, n_day, k, start_date, end_date)
    upper_bollinger = upperBollinger(df, col, n_day, k, start_date, end_date)
    plotPrice(audjpy_df, timeStampToStr(dates.iloc[n_day]),  timeStampToStr(dates.iloc[-1]))
    plt.plot(dates.iloc[n_day:], upper_bollinger, color='red')
    plt.plot(dates.iloc[n_day:], lower_bollinger, color='red')
    plt.show()


bollingerBands(audjpy_df, 'PX_LAST', 20, 1, '2018-01-01', '2018-06-01')


# In[262]:


"""
Plots rectracement level prices and Bollinger bands
"""
def plotRetracementBollinger(df, col, n_day, k, start_date, end_date, w = np.linspace(1,10), c='r'):
    
    sub_df = dataframeBetween(df, start_date, end_date)
    sub_df_shift_n = sub_df[n_day:]
    column = sub_df_shift_n[col]
    dates = sub_df_shift_n['Dates']
   
    
    d1 = timeStampToStr(dates.iloc[0])
    d2 = timeStampToStr(dates.iloc[-1])
    
    plotPrice(sub_df_shift_n, d1, d2)
    
    for i in range(len(column)):
        price = column.iloc[i]
    
        if isRetracementLevel(sub_df_shift_n, col, price, start_date, end_date, w):
            plt.plot(dates.iloc[i], price, marker='o', markersize=6, color=c)
    
    d1 = timeStampToStr(sub_df['Dates'].iloc[0])
    d2 = timeStampToStr(sub_df['Dates'].iloc[-1])

    lower_bollinger = lowerBollinger(sub_df, col, n_day, k, d1 , d2)
    upper_bollinger = upperBollinger(sub_df, col, n_day, k, d1, d2)

    plt.plot(dates, upper_bollinger, color='red')
    plt.plot(dates, lower_bollinger, color='red')
    plt.show()
    

plotRetracementBollinger(audjpy_df, 'PX_LAST', 90, 1, '2015-6-13', '2016-06-17')


# In[ ]:


audjpy_df = pd.read_excel('/Users/Ian/Downloads/AUDJPY_Data.xlsx')


# In[356]:


"""
Returns a future date equal to date incremented by the specified number of years, 
months, and days

Parameter date:
"""
def deltaTime(date, num_yr = 0, num_m=0, num_d=0):

    future_date = toTimestamp(date) + pd.DateOffset(years = num_yr, months = num_m , days = num_d)
    year = str(future_date.year)
    month = str(future_date.month)
    day = str(future_date.day)
    
    if len(month) < 2:
        month = '0' + month
    if len(day) < 2:
        day = '0' + day
    
    return year + '-' + month + '-' + day

"""
Returns a a list of two-tuples containing the start date and
an end-date of the backtest windows

Parameter df: the backtest data
Precondition df is a pandas dataframe
"""
def generateBacktest(df, col, n_day, k, period, sample_size):
    dates = df['Dates']
    sample_windows = {}
    
    for date in range(sample_size):
  
        index = random.randint(n_day, len(df[col]))
        
        d1 = timeStampToStr(dates.iloc[index])
        # 3 month shift into the future
        d2 = deltaTime(d1, 0, period, 0)
        
        sub_df = dataframeBetween(df, d1, d2)
        prices = sub_df[col]
        signals = []
        
        ub = upperBollinger(df, col, n_day, k, d1, d2)
        lb = upperBollinger(df, col, n_day, k, d1, d2)
        
        for i in range(len(prices) - n_day):
            price = prices.iloc[i]
            isRetracement = isRetracementLevel(df, col, price, d1, d2, w=np.linspace(1,10))
            # BUY signal
            if isRetracement and price <= lb[i]:
                signals.append(-price)
            # SELL signal
            elif isRetracement and price > ub[i]:
                signals.append(price)
        # prevents a BUY from appearing as a net loss in the net_return calculation
        net_returns = []
        num_buys = 0
        buy_prices = []
        for signal_i in range(len(signals)):
            if signals[signal_i] < 0:
                num_buys += 1
                buy_prices.append(-1 * signals[signal_i])   
            else :
                # cannot SELL before BUY
                if num_buys != 0:
          
                    net_returns.append(num_buys * signals[signal_i] - sum(buy_prices))
                    buy_prices = []
                    num_buys = 0
        print(signals)
       
        sample_windows[d1+ " " + d2] = float('{0:.3f}'.format(sum(net_returns)))
    return sample_windows
        
        
generateBacktest(audjpy_df, 'PX_LAST', 10, 1, 3, 5)


# In[353]:


moving_avgs = {'5-day' : None, '10-day' : None, '20-day' : None,
               '30-day' : None,'40-day' : None, '50-day' : None,
               '75-day' : None, '100-day' : None, '200-day' : None} 

sample_size = 3

for key in moving_avgs.keys():
    days_in_avg = int(key.split('-')[0])
    date_return_dict = generateBacktest(audjpy_df, 'PX_LAST', days_in_avg, 1, sample_size)
    print(str(days_in_avg) + " sample size " + str(sample_size) + " test returns: %s" % sum(date_return_dict.values()))
    


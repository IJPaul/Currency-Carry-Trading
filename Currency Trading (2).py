
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[3]:


import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
import math
from datetime import datetime
from datetime import timedelta


# In[4]:


moving_avgs = {'5-day' : None, '10-day' : None, '20-day' : None,
               '30-day' : None,'40-day' : None, '50-day' : None,
               '75-day' : None, '100-day' : None, '200-day' : None} 


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


# In[10]:


audjpy_df = pd.read_excel('/Users/Ian/Downloads/AUDJPY_Data.xlsx')
col = audjpy_df['PX_LAST']
dataframeBetween(audjpy_df, "2017-10-30", "2017-10-31")


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


# In[13]:


"""
Returns a new dataframe containing
every Nth row of the passed dataframe df. 
Starts at the first row of the df by default.
"""
def getEveryNthRow(df, n, start=0):
    return df.iloc[start::n, :]


# In[329]:


audjpy_df = pd.read_excel('/Users/Ian/Downloads/AUDJPY_Data.xlsx')
#col = audjpy_dfhttp://localhost:8888/notebooks/Currency%20Trading.ipynb#['PX_LAST']
print(EMA(audjpy_df, 'PX_LAST', 90, "2018-04-07", "2018-11-01"))
plt.plot(range(len(EMA(audjpy_df, 'PX_LAST', 100, "2016-10-07", "2018-11-01"))), EMA(audjpy_df, 'PX_LAST', 100, "2016-10-07", "2018-11-01"))
#EMA(audjpy_df, 'PX_LAST', 50, "2016-01-04", "2018-11-01")


# In[103]:



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
    
plotPrice(audjpy_df, "2017-10-15", "2018-11-15")


# In[21]:


"""
Returns a list of the peaks in df[col] between
start date and end_date in df[col]. Smaller W 
values correspond to the width of peaks. Local maxima
will have a smaller W and global maxima will have larger W

Parameter w: w is the list of peak widths to search for
Precondition: w is a 1-D array of numbers
"""
def findPeaks(df, col, start_date, end_date, w):
    sdf = dataframeBetween(df, start_date, end_date)
    column = sdf[col].values
    return column[signal.find_peaks_cwt(column, w)]


# In[22]:


"""
Returns a list of the troughs in df[col] between
start_date and end_date in df[col]. Smaller W 
values correspond to the width of peaks. Local maxima
will have a smaller W and global maxima will have larger W

Parameter w: w is the list of peak widths to search for
Precondition: w is a 1-D array of numbers
"""
def findTroughs(df, col, start_date, end_date, w):
    sdf = dataframeBetween(df, start_date, end_date)
    column = (-1 * sdf[col]).values
    return -1* np.array(column[signal.find_peaks_cwt(column, w)])


# In[53]:


data = audjpy_df['PX_LAST']
#print(findPeaks(audjpy_df, 'PX_LAST', '2016-12-01', '2018-12-21', np.arange(1,100)))
#print(findTroughs(audjpy_df, 'PX_LAST', '2016-12-01', '2016-12-21', [4,10]))
def plotPeaksTroughs(df, col, start_date, end_date, w, pc = 'g', tc = 'r'):
    plotPrice(df, start_date, end_date)
    for peak in findPeaks(df, col, start_date, end_date, w):
        plt.axhline(y=peak , color=pc, linestyle='-')
    for trough in findTroughs(df, col, start_date, end_date, w):
        plt.axhline(y=trough , color=tc, linestyle='-')

plotPeaksTroughs(audjpy_df, 'PX_LAST', '2017-12-01', '2018-05-21')


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


# In[86]:


def plotRetracementLevels(df, col, start_date, end_date, w = np.linspace(1,10,1), c = 'r'):
    plotPrice(df, start_date, end_date)
    sub_df = dataframeBetween(df, start_date, end_date)
    column = sub_df[col]
    
    for i in range(len(column)):
        #i = sub_df.loc[sub_df[col] == price].index[0]
        price = column.iloc[i]
        if isRetracementLevel(sub_df, col, price, start_date, end_date, w):
            #plt.axhline(y=price, color=c, linestyle='-')
            plt.plot(sub_df['Dates'].iloc[i], price, marker='o', markersize=6, color=c)
            
column = dataframeBetween(audjpy_df, '2016-12-01', '2016-12-21')['PX_LAST']
plotRetracementLevels(audjpy_df, 'PX_LAST', '2017-12-01', '2018-07-21')


# In[38]:


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
    n_day_arr = sdf[col][:n_day]
    initial_ema = sum(n_day_arr) / n_day
    ema_arr.append(initial_ema)
  
    for price in sdf[col][n_day : len(sdf[col])]:    
        ema_arr.append((price - ema_arr[-1]) * weighting_mult + ema_arr[-1])
    return ema_arr


# In[39]:


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

sma = EMA(audjpy_df, 'PX_LAST', 5, '2017-10-01', '2017-11-05')
print(len(sma))
plotPrice(audjpy_df, '2017-10-01', '2017-11-05')
x = ndayStdDev(audjpy_df, 'PX_LAST', 5, '2017-10-01', '2017-11-05')
print(x)
print("number of std dev data points %s" % len(x))
print("number of price data points %s" % len(dataframeBetween(audjpy_df, '2017-10-01', '2017-11-05')))


# In[98]:


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
    sma = EMA(df, col, n_day, start_date, end_date)
    sdf = dataframeBetween(df, start_date, end_date)
    ub = []
    for i in range(len(sdf[col]) - n_day):
        ub.append(sma[i] + k*((ndayStdDev(df, col, n_day, start_date, end_date))[i]) ) 
    return ub


# In[99]:


"""
Returns the lower bollinger band array of prices where each element is the N period EMA - Kσ up to and including
price. σ is the N period standard deviation in price.
"""
def lowerBollinger(df, col, n_day, k, start_date, end_date):
    sma = EMA(df, col, n_day, start_date, end_date)
    sdf = dataframeBetween(df, start_date, end_date)
    lb = []
    for i in range(len(sdf[col]) - n_day):
        lb.append(sma[i] - k*((ndayStdDev(df, col, n_day, start_date, end_date))[i]))
    return lb


# In[100]:


def timeStampToStr(t):
   
    year = str(t.year)
    month = str(t.month)
    day = str(t.day)
    return year + '-' + month + '-' + day


# In[101]:


"""
Plots Bollinger Bands of df[col] between start_date and end_date
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


bollingerBands(audjpy_df, 'PX_LAST', 20, 1, '2018-05-01', '2018-11-05')


# In[111]:


def plotRetracementBollinger(df, col, n_day, k, start_date, end_date, w = np.linspace(1,10,1), c='r'):
    
    sub_df = dataframeBetween(df, start_date, end_date)
    sub_df_shift_n = sub_df[n_day:]
    column = sub_df_shift_n[col]
    dates = sub_df_shift_n['Dates']
    print(len(dates))
    
    d1 = timeStampToStr(dates.iloc[0])
    d2 = timeStampToStr(dates.iloc[-1])
    
    plotPrice(sub_df_shift_n, d1, d2)
    
    for i in range(len(column)):
        price = column.iloc[i]
        # Consider using d1 and d2 for dates
        if isRetracementLevel(sub_df_shift_n, col, price, start_date, end_date, w):
            plt.plot(dates.iloc[i], price, marker='o', markersize=6, color=c)
    
    d1 = timeStampToStr(sub_df['Dates'].iloc[0])
    d2 = timeStampToStr(sub_df['Dates'].iloc[-1])

    lower_bollinger = lowerBollinger(sub_df, col, n_day, k, d1 , d2)
    upper_bollinger = upperBollinger(sub_df, col, n_day, k, d1, d2)

    plt.plot(dates, upper_bollinger, color='red')
    plt.plot(dates, lower_bollinger, color='red')
    plt.show()
    

plotRetracementBollinger(audjpy_df, 'PX_LAST', 20, 1, '2016-11-01', '2017-04-25')


# In[61]:


"""
Returns True if it is profitable to buy the currency pair
at the specified price
"""
def isBuy(price):
    return None


# In[ ]:


audjpy_df = pd.read_excel('/Users/Ian/Downloads/AUDJPY_Data.xlsx')
col = audjpy_df['PX_LAST']
EMA(audjpy_df, col, 50, '2016-12-01', '2018-03-21')
#audjpy_df


# In[ ]:


def incrementDate(df, date, num_yr=0, num_m=0, num_d=0):
    year = int(datetime.now().year)
    month = int(datetime.now().month)
    day = int(datetime.now().day)
    

"""
Returns a a list of two-tuples containing the start date and
an end-date of the backtest windows

Parameter df: the backtest data
Precondition df is a pandas dataframe
"""
def generateBacktest(df, start_date, end_date):
    year = int(start_date.year)
    month = int(start_date.year)
    
    


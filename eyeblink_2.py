# %% [markdown]
# # EYE BLINK DETECTION:
# # 2) Adaptive Model
# For this part, we will experiment on the variables
# 
# > EAR_THRESHOLD = 0.21 # eye aspect ratio to indicate blink  
# > EAR_CONSEC_FRAMES = 3 # number of consecutive frames the eye must be below the threshold  
# > SKIP_FIRST_FRAMES = 0 # how many frames we should skip at the beggining  
# 
# to make them adaptive to subject of the video. We will basically ***normalize*** across samples of a subject.
# 
# **NOT:** Starting from this section, we will use utils.py script to import necessary function that we implement on previous notebooks.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.073924Z","iopub.execute_input":"2023-06-16T06:06:30.074557Z","iopub.status.idle":"2023-06-16T06:06:30.086730Z","shell.execute_reply.started":"2023-06-16T06:06:30.074475Z","shell.execute_reply":"2023-06-16T06:06:30.085393Z"}}
# import packages
import scipy.stats as st
from IPython.display import YouTubeVideo
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import gridspec
%matplotlib inline

# import utility functions
from utils_eyeblink import *

# define three constants. 
# You can later experiment with these constants by changing them to adaptive variables.
EAR_THRESHOLD = 0.21 # eye aspect ratio to indicate blink
EAR_CONSEC_FRAMES = 3 # number of consecutive frames the eye must be below the threshold
SKIP_FIRST_FRAMES = 0 # how many frames we should skip at the beggining

# initialize output structures
scores_string = ""

# %% [markdown]
# **Load outputs of basic model:**

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.089856Z","iopub.execute_input":"2023-06-16T06:06:30.090489Z","iopub.status.idle":"2023-06-16T06:06:30.112412Z","shell.execute_reply.started":"2023-06-16T06:06:30.090391Z","shell.execute_reply":"2023-06-16T06:06:30.110405Z"}}
#define path and dataset_name
path = "../input/eye-blink-detection-1-simple-model"
dataset_name = "talking"

# load datasets
c_pred, b_pred, df, c_test, b_test, s_str = load_datasets(path, dataset_name)

# check results
print(np.array(c_pred).shape, np.array(b_pred).shape)
print(np.array(c_test).shape, np.array(b_test).shape)
print()

#display statistics
print(s_str)

#display the first rows of data frame
df[:3]

# %% [markdown]
# # Analyzing Data
# 
# After seeing the results of basic model, now we can start **implementing adaptive thresholds model**.
# 
# Let's take a glance at our data by plotting it.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.215339Z","iopub.execute_input":"2023-06-16T06:06:30.215868Z","iopub.status.idle":"2023-06-16T06:06:30.239560Z","shell.execute_reply.started":"2023-06-16T06:06:30.215813Z","shell.execute_reply":"2023-06-16T06:06:30.236937Z"}}
# plot whole data 
data = np.array(df['avg_ear'])
plt.figure(figsize=(15,5))
plt.xticks(np.arange(0, len(data)+1, 200))
m,b = np.polyfit(np.arange(len(data)),data, 1)
print("SLOPE = {:.5f}".format(m))
plt.plot(np.arange(len(data)),data,'yo', np.arange(len(data)), m*np.arange(len(data))+b, '--r')
plt.plot(data);

# %% [markdown]
# Remark that slope of linear regression is ideally m >= 0. Because blinking occurs rarely and they don't impact much to the general trend of ear values. 
# 
# If we get first 100 ear values and plot them with **linear fitting** what it would be like?

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.241172Z","iopub.status.idle":"2023-06-16T06:06:30.242063Z"}}
#get first n frames and plot it with linear regression
n = 100
data100 = data[:n]
m,b = np.polyfit(np.arange(n),data100, 1)
print("SCOPE = {:.5f}".format(m))
plt.plot(np.arange(n),data100,'yo', np.arange(n), m*np.arange(n)+b, '--g');

# %% [markdown]
# So getting first n = 100 frames doesn't catch a full blink behaviour, ending parts are missing. To get a proper blink, we need to estimate a minimum frame count truly. 
# 
# Remark that, the linear regression above has negative slope. So we need to iterate on number of n (starting with some minimum treshold like 50) until we get a positive slope just like the behaviour of full dataset above. Also if slope is converges around some proper value it will be a good sign for proper n. This way we will be sure on that we caught a full blink.

# %% [markdown]
# # Adaptive Thresholds: SKIP_FIRST_FRAMES
# 
# So we define a function to estimate calibration hyperparameter n.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.243578Z","iopub.status.idle":"2023-06-16T06:06:30.244300Z"}}

def estimate_first_n(data, start_n=50, limit_n=300, step=1, epsilon=10e-8):
    n = start_n 
    while True:
        # for first n values fit a linear regression line 
        data0 = data[:n]
        m0,b0 = np.polyfit(np.arange(n),data0, 1)
        
        # check if n + step reaches limit
        if n + step > limit_n-1:
            print("error - reached the limit")
            break

        # for first n + step values fit a linear regression line 
        data1 = data[:n+step]
        m1,b1 = np.polyfit(np.arange(n + step),data1, 1)

        # if m1-m0 converges to epsilon
        if abs(m1 - m0) < epsilon and m0 > 0:
            break
        n += step
        
    return n, m0

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.245537Z","iopub.status.idle":"2023-06-16T06:06:30.246200Z"}}
# calculate a limit_n for 10 secs with using fps of the video
fps = int(s_str[49:51])
limit_secs = 10
limit_frame = limit_secs * fps

# run the function above
n,m0 = estimate_first_n(data, limit_n=limit_frame)

# get first n frames and plot it with linear regression
calibration = data[:n]
m,b = np.polyfit(np.arange(n), calibration, 1)
linear = m*np.arange(n)+b
print("n = {:.5f}".format(n))
print("m = {:.5f}".format(m))
plt.figure(figsize=(15,5))
plt.xticks(np.arange(0, n, 5))
plt.grid()
plt.plot(np.arange(n),calibration,'yo', np.arange(n), linear, '--g');

# %% [markdown]
# We can see that our function can detect some full blinks. Actually this function waits a blink to finish it's action, but doesn't guarantee to detect at least one blink. There is a possibility that if driver doesn't blink for a long time and his ears just decrease without any blink in initial period, algorithm doesn't return an error.
# 
# Also there could be another problem, calibration phase may exceed limit_n = 10 secs for most cases. So we need to find an adaptive way to use limit_n and run the function until it gets **at least one blink**.
# 
# **NOTE TO MYSELF:** research casual impact for this case https://github.com/dafiti/causalimpact

# %% [markdown]
# # Another Problem: Facial Actions (Smiling, Yawning etc.)
# 
# Let's go back to the data analysis. As you remember, we have detected 2 blinks.
# 
# First, the blink which we detected between 30th and 90th frames. With enlarged tails it can be anywhere between 0-163. Second, the blink between 138-218.
# 
# Plot both of them to analyze seperately.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.247521Z","iopub.status.idle":"2023-06-16T06:06:30.248354Z"}}
# plot data between 0 - 163
blink1 = calibration[:163]
plt.figure(figsize=(15,5))
plt.xticks(np.arange(0, len(blink1), 5))
plt.grid()
plt.plot(blink1);

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.249664Z","iopub.status.idle":"2023-06-16T06:06:30.250352Z"}}
# plot data between 138 - 218
blink2 = calibration[138:]
plt.figure(figsize=(15,5))
plt.xticks(np.arange(0, len(blink2), 2),np.arange(138, 138+len(blink2),2))
plt.grid()
plt.plot(blink2);

# %% [markdown]
# If we examine two blinks with more precise ranges and get frame samples by running simple_model, we can display them to check their nature.
# * blink1 -> (56,87)
# * blink2 -> (165,178)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.251647Z","iopub.status.idle":"2023-06-16T06:06:30.252603Z"}}
# run simple model to get frames list
_, _, _, fr, _, _ = simple_model("../input/blinkdata/talkingFace/talking.avi");

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.253745Z","iopub.status.idle":"2023-06-16T06:06:30.254377Z"}}
# display starting, middle and ending frames of two blinks
display_blinks([(56,87),(165,178)], fr)

# %% [markdown]
# It turns out that blink1 is a smiling face and arguably not a proper blink. On the other hand blink2 is a normal blink. We will use this information for later experiments when we need a normal blink.
# 
# We can double check this by looking at blink prediction and blink test data from of simple model.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.357010Z","iopub.execute_input":"2023-06-16T06:06:30.357529Z","iopub.status.idle":"2023-06-16T06:06:30.374618Z","shell.execute_reply.started":"2023-06-16T06:06:30.357455Z","shell.execute_reply":"2023-06-16T06:06:30.372753Z"}}
# print first 5 blinks of test data
print(b_test[0:5])

# print first 5 blinks of prediction data
print(b_pred[0:5])

# %% [markdown]
# We can see that simpe model predicts blink1(56,87) as a blink when annotations don't. So it's a false-positive sample. Moreover blink2(165,178) is catched by both.
# 
# Consequently facial expressions like smiling and ywaning are drawbacks of both simple and adaptive models. Btw machine learning approaches can be good solution here but we need a practical solution which can be added to these two models for now. We will implement SVM and deep learning models in later notebooks.
# 
# **NOTE TO MYSELF:** Research ARIMA and correlation analysis for this case.

# %% [markdown]
# # Adaptive Thresholds: EAR_THRESHOLD
# While continuing to build adaptive model, we can try something with error plots:

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.376139Z","iopub.status.idle":"2023-06-16T06:06:30.376950Z"}}
# calculate errors
errors = calibration - linear

# plot errors
plt.figure(figsize=(15,5))
plt.xticks(np.arange(0, n, 5))
plt.grid()
plt.plot(calibration,'yo', linear, '--g', errors, '--b')
plt.legend(labels = ['ear', 'linear fitting', 'error'])
plt.show()

# cumulative errors
cum_errors = errors.cumsum()

# plot cumulative errors
plt.figure(figsize=(15,5))
plt.xticks(np.arange(0, n, 5))
plt.grid()
plt.plot(np.arange(n), cum_errors, '--k')
plt.legend(labels = ['cumulative error'])
plt.show();

# %% [markdown]
# Cumulative error doesn't say much on blinks since it's effect delayed. But normal error can be used for anomalie detection.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.378858Z","iopub.status.idle":"2023-06-16T06:06:30.379673Z"}}
# plot histogram of calibration and errors
plt.hist(calibration, bins=30)
plt.hist(errors, bins=30)
plt.legend(['ear', 'error']);

# %% [markdown]
# Also errors more like normally distributes than ear values so can use some statistical tools on it.  

# %% [markdown]
# **First we can try outlier detection with IRQ method. Let's define a function that removes outliers from a list:**

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.381425Z","iopub.status.idle":"2023-06-16T06:06:30.382122Z"}}
# remove outliers from a list by using IQR method
def detect_outliers_iqr(input_list):
    # calculate interquartile range
    q25, q75 = np.percentile(input_list, 25), np.percentile(input_list, 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    
    # identify outliers
    outliers = [(i, x) for (i, x) in list(enumerate(input_list)) if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    
    # remove outliers
    clean_input_list = [x for x in input_list if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(clean_input_list))
    print("")
    
    return clean_input_list, outliers, upper, lower

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.384264Z","iopub.status.idle":"2023-06-16T06:06:30.384952Z"}}
# run detect_outliers_iqr() on errors
_, outliers, upper, lower =  detect_outliers_iqr(errors)
outlier_indexes = list(zip(*outliers))[0]
outlier_values = list(zip(*outliers))[1]

# run detect_outliers_iqr() on calibration
_, outliers_c, upper_c, lower_c =  detect_outliers_iqr(calibration)
outlier_indexes_c = list(zip(*outliers_c))[0]
outlier_values_c = list(zip(*outliers_c))[1]

# %% [markdown]
# **Also we can try confidence interval method:**

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.386325Z","iopub.status.idle":"2023-06-16T06:06:30.386984Z"}}
# calculate upper and lower limits for given confidence interval
def detect_outliers_conf(input_list, confidence=0.95):
    # identify boudaries
    lower, upper  = st.t.interval(confidence, len(input_list)-1, loc=np.mean(input_list), \
                                        scale=st.sem(input_list))
    
    # identify outliers
    outliers = [(i, x) for (i, x) in list(enumerate(input_list)) if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    
    # remove outliers
    clean_input_list = [x for x in input_list if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(clean_input_list))
    print("")
    
    return clean_input_list, outliers, upper, lower

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.388529Z","iopub.status.idle":"2023-06-16T06:06:30.389214Z"}}
# run detect_outliers_conf() for 0.99 on errors
_, _, _, lower_conf = detect_outliers_conf(errors, 0.99)

# run detect_outliers_conf() for 0.99 on calibration
_, _, _, lower_conf_c = detect_outliers_conf(calibration, 0.99)

# %% [markdown]
# Another outliner detection using z_scores.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.390377Z","iopub.status.idle":"2023-06-16T06:06:30.390985Z"}}
# calculate upper and lower limits for given confidence interval
def detect_outliers_z(input_list, z_limit=2):
    # identify boudaries
    mu = input_list.mean()
    sigma = input_list.std()
    val = z_limit * sigma
    lower  =  mu - val
    upper =  mu + val
    
    # identify outliers
    outliers = [(i, x) for (i, x) in list(enumerate(input_list)) if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    
    # remove outliers
    clean_input_list = [x for x in input_list if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(clean_input_list))
    print("")
    
    return clean_input_list, outliers, upper, lower

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.493707Z","iopub.execute_input":"2023-06-16T06:06:30.494447Z","iopub.status.idle":"2023-06-16T06:06:30.518731Z","shell.execute_reply.started":"2023-06-16T06:06:30.494121Z","shell.execute_reply":"2023-06-16T06:06:30.516585Z"}}
# run detect_outliers_z() for z_limit=2 on errors
_, outliers_z, _, lower_z = detect_outliers_z(errors, 2)
outlier_indexes_z = list(zip(*outliers_z))[0]
outlier_values_z = list(zip(*outliers_z))[1]

# run detect_outliers_z() for z_limit=2 on calibration
_, outliers_z_c, _, lower_z_c = detect_outliers_z(calibration, 2)
outlier_indexes_z_c = list(zip(*outliers_z_c))[0]
outlier_values_z_c = list(zip(*outliers_z_c))[1]

# %% [markdown]
# Put them all together and plot:

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.520305Z","iopub.status.idle":"2023-06-16T06:06:30.521539Z"}}
# plot calibration and errors
plt.figure(figsize=(15,5))
plt.xticks(np.arange(0, n, 5))
plt.grid()
plt.plot(calibration,'yo', linear, '--g', errors, '--b')
plt.legend(labels = ['ear', 'linear fitting', 'error'], loc='lower right')

# plot outliers for errors
plt.plot(outlier_indexes, outlier_values, 'ro')
plt.hlines(lower, 0, 218, colors='r', linestyles='-')
plt.text(0, lower + 0.01, 'iqr lower bound = {:.4f}'.format(lower), color='r')

# plot outliers for calibration
plt.plot(outlier_indexes_c, outlier_values_c, 'ro')
plt.hlines(lower_c, 0, 218, colors='r', linestyles='-')
plt.text(0, lower_c - 0.02, 'iqr lower bound = {:.4f}'.format(lower_c), color='r')

# 99% confidence for errors
plt.hlines(lower_conf, 0, 218, colors='c', linestyles='-')
plt.text(0, lower_conf + 0.01, '0.99 conf lower bound = {:.4f}'.format(lower_conf), color='c')

# 99% confidence for calibration
plt.hlines(lower_conf_c, 0, 218, colors='c', linestyles='-')
plt.text(0, lower_conf_c + 0.01, '0.99 conf lower bound = {:.4f}'.format(lower_conf_c), color='c')

# z_limit=2 for errors
plt.plot(outlier_indexes_z, outlier_values_z, 'kx')
plt.hlines(lower_z, 0, 218, colors='k', linestyles='-')
plt.text(0, lower_z - 0.02, 'z_limit=2 lower bound = {:.4f}'.format(lower_z), color='k')

# z_limit=2 for calibration
plt.plot(outlier_indexes_z_c, outlier_values_z_c, 'kx')
plt.hlines(lower_z_c, 0, 218, colors='k', linestyles='-')
plt.text(0, lower_z_c + 0.01, 'z_limit=2 lower bound = {:.4f}'.format(lower_z_c), color='k')
plt.show()

# %% [markdown]
# .99 confidence interval didin't produce significant results. 
# 
# And running **detect_outliers_iqr()** on errors gives much better results than running on calibration since it's lower bound is better for estimating blinks. 
# 
# * lower bound of calibration --> 0.1781 (ear value). Detects 8 frames (6 for blink1, 2 for blink2)
# * lower bound of errors --> -0.0749 (error value). Detects 16 frames (12 for blink1, 4 for blink2)
# 
# Also analyzing with **z_limit =2** on calibration does better than running on errors.
# * lower bound of calibration --> 0.1781 (ear value). Detects 12 frames (9 for blink1, 3 for blink2)
# * lower bound of errors --> -0.0749 (error value). Detects 13 frames (9 for blink1, 4 for blink2)
# 
# So we can build a pipeline for calibration phase to returns EAR_THRESHOLD in an adaptive way.

# %% [markdown]
# # Adaptive Thresholds: EAR_CONSEC_FRAMES
# 

# %% [markdown]
# In this case autocorrelation and partial-autocorrelation plots may be useful.
# 
# To understand how both of them work, you can read this:  
# https://towardsdatascience.com/time-series-forecasting-arima-models-7f221e9eee06
# 
# And you can watch this video:

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.522769Z","iopub.status.idle":"2023-06-16T06:06:30.523551Z"}}
# stream the youtube video
YouTubeVideo('ZjaBn93YPWo', width=720)

# %% [markdown]
# Then plot autocorrelation and partial-autocorrelation to see how a frame is correlated to previous frames.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.525216Z","iopub.status.idle":"2023-06-16T06:06:30.525886Z"}}
# plot autocorrelation and partial-autocorrelation of WHOLE DATASET
plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2) 
ax0 = plt.subplot(gs[0,:])
ax0.set_title('Whole Dataset')
plt.xticks(np.arange(0, len(data)+1,200))
plt.plot(data)
plt.grid()
ax1 = plt.subplot(gs[1,0])
plt.xticks(np.arange(0, 1000+1,100))
plot_acf(data, lags=1000, ax=ax1)
plt.grid()
ax2 = plt.subplot(gs[1,1])
plt.xticks(np.arange(0, 30+1,2))
plot_pacf(data, lags=30, ax=ax2)
plt.grid();

# %% [markdown]
# If we analyze ACF of whole data, we can infer that this is a non-statinary time series but initial phase takes too long to decrease until 65 (during this part plot behaves like stationary data), then increase again until 200 and drops quickly (this part behaves like non-stationary). It's somehow correlated with first two blinks.
# 
# > Reminder:  
# > blink1 -> (56,87)  
# > blink2 -> (165,178)
# 
# And for PACF plot numbers of 4, 7, 10, 15, 19 are significant.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.628703Z","iopub.execute_input":"2023-06-16T06:06:30.629249Z","iopub.status.idle":"2023-06-16T06:06:30.654223Z","shell.execute_reply.started":"2023-06-16T06:06:30.629181Z","shell.execute_reply":"2023-06-16T06:06:30.652415Z"}}
# plot autocorrelation and partial-autocorrelation of CALIBRATION SET
plt.figure(figsize=(17, 10))
gs = gridspec.GridSpec(2, 2) 
ax0 = plt.subplot(gs[0,:])
ax0.set_title('Calibration Set')
plt.xticks(np.arange(0, len(calibration)+5,5))
plt.grid()
plt.plot(calibration)
ax1 = plt.subplot(gs[1,0])
plt.grid()
plt.xticks(np.arange(0, 217+1,10))
plot_acf(calibration, lags=217, ax=ax1)
ax2 = plt.subplot(gs[1,1])
plt.grid()
plt.xticks(np.arange(0, 30+1,2))
plot_pacf(calibration, lags=30, ax=ax2);

# %% [markdown]
# When we focus on ACF of calibration set (first 217 frames), we can easily say all plot is non-stationary. In case of effects of blinks, it's hard to say anything on significant manner but after finishing of blink1 (56-87), we see that trend of increasing correlation of lags starts to decrease again. This may be the effect of blink1. After effects of blink1 disappears, graph again starts to become correlated with previous points. And when blink2 (165-178) occurs all of balance in correlation dissappears and converges to zero.
# 
# For PACF 5, 21 and 29 are significant numbers. Since we just two blinks in calibration phase we may think about this number somehow relevant to two blinks.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.655893Z","iopub.status.idle":"2023-06-16T06:06:30.656957Z"}}
# plot autocorrelation and partial-autocorrelation of BLINK1
plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2) 
ax0 = plt.subplot(gs[0,:])
ax0.set_title('Tailed Blink1')
plt.xticks(np.arange(0, len(blink1)+3,5))
plt.grid()
plt.plot(blink1)
ax1 = plt.subplot(gs[1,0])
plt.grid()
plt.xticks(np.arange(0, 163+1,10))
plot_acf(blink1, lags=len(blink1)-1, ax=ax1)
ax2 = plt.subplot(gs[1,1])
plt.grid()
plt.xticks(np.arange(0, 30+1,2))
plot_pacf(blink1, lags=30, ax=ax2);

# %% [markdown]
# This set (0-163) is tailed version of blink1 (56-87) and behaves like perfect non-stationary data as expected.
# 
# For PACF significant numbers are 4, 18, 22, 26. 
# 
# Generally a basic EAR value series, follows a linear regression trend (m~0, let's call this **open-eye line**) but with a sudden major drops when blink occurs. Points on **open-eye line** highly correlated to each other. But not to the points on blinks. Even though there's a inner correlation between points on blink points, their correlation decreases when the lag reaches start and end points of blinks. So, we can say that significant number on PACF is relevant to duration of blinks, especially first significant number=4 for this case.
# 
# But there's also a nuance for blink points. We say that they are correlated to each other, actually this is only true for seperate halves. After reaching bottom point of blink, there's also decrease in correlation, so number=4 may be relevant to duration/2, not the whole duration.
# 
# We can chect this by looking at blink1 and blink2.
# * if we more focus on (56-76) and try to capture blink1 action more precisely, b duration = 76-56 = 20, it's somewhere between first significant number 4*2=8 and other significant numbers = 18, 22, 26
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:06:30.658676Z","iopub.status.idle":"2023-06-16T06:06:30.659546Z"}}
# plot autocorrelation and partial-autocorrelation of BLINK2
plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2) 
ax0 = plt.subplot(gs[0,:])
ax0.set_title('Tailed Blink2')
plt.xticks(np.arange(0, len(blink2)+1,2))
plt.grid()
plt.plot(blink2)
ax1 = plt.subplot(gs[1,0])
plt.grid()
plt.xticks(np.arange(0, 80+1,5))
plot_acf(blink2, lags=len(blink2)-1, ax=ax1)
ax2 = plt.subplot(gs[1,1])
plt.grid()
plt.xticks(np.arange(0, 30+1,2))
plot_pacf(blink2, lags=30, ax=ax2);

# %% [markdown]
# The same analysis above,
# 
# * major drop for blink2=(30-37) and duration = 37-30 = 7. 
# * significant numbers on PACF are 4, 17 and others.
# 
# We can conclude that duration of blink2 somewhere between first significant number*2 and second significant number, just like the blink1.
# 
# So this way, we may use first significant number as adaptive threshold of EAR_CONSEC_FRAMES.

# %% [markdown]
# **NOTE TO MYSELF:**
# 
# Further research: 
# * Improve estimate_first_n() function. Research the possibility of using CasualImpact (https://github.com/dafiti/causalimpact) for adaptive SKIP_FIRST_FRAME
# * ARIMA and correlation analysis for detecting real blinks (not other facial expressions like yawning and smiling)
# * Finish the pipeline that uses proper anomaly detection method for the case of adaptive EAR_THRESHOLD
# * Implement a function to find significant numbers in PACFL to use for adaptive EAR_CONSEC_FRAMES
# * Make Normalizaton across videos of subject by using calibration phase. Research non-Gaussian techniques like box-cox transformation
# 
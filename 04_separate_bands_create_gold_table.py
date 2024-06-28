# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview - Separate Bands and Create Gold Table 
# MAGIC
# MAGIC Applying a second-order Butterworth Filter to EEG signals allows you to isolate specific frequency components while reducing the strength or amplitude of others, thereby enabling certain frequency components to pass through relatively unaffected. This process aids in extracting meaningful information from EEG data, thereby enhancing the analysis of brain activity across various frequency bands.
# MAGIC
# MAGIC The EEG signals from each channel underwent filtering using a second-order Butterworth Filter across distinct physiological frequency ranges: 2–4 Hz (delta), 4.5–7.5 Hz (theta), 8–12.5 Hz (alpha), 13–30 Hz (beta), and 30–45 Hz (gamma).
# MAGIC
# MAGIC ##### In this notebook we will:
# MAGIC   * Isolate specific EEG frequencies using a second-order Butterworth Filter.
# MAGIC   * Explore dataset based on summarize statistics. 
# MAGIC   * Examine statistical metrics utilizing Databricks' integrated commands `describe` and `summary` for potential future data manipulation.
# MAGIC     * `describe`: Provides statistics including count, mean, standard deviation, minimum, and maximum.
# MAGIC     * `summary`: describe + interquartile range (IQR)
# MAGIC     * Look for missing or null values in the data that will cause outliers or skew the data, zero values are expected with EEG data so no need to wrangle them.
# MAGIC   * Create Gold Layer Table
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Retrieve data from Silver Tables and convert in place PySpark Dataframe to Pandas Dataframe

# COMMAND ----------

# REST Reference Method Table
df_rest_ref = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_rest_ref_silver ORDER BY index_id ASC""").toPandas()
# Average Reference Method Table
df_avg_ref = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_avg_ref_silver ORDER BY index_id ASC""").toPandas()

# Show the result
# display(df_rest_ref)
# df_rest_ref.show()

# Show the result
# display(df_avg_ref)
# df_avg_ref.show()

# Select distinct values from the 'patient_id' column
patient_ids = df_rest_ref["patient_id"].unique()

# Display distinct values
print(f"patient_ids:::{patient_ids}")

# Display DataFrames
display(df_rest_ref)
display(df_avg_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Isolate specific EEG frequencies using a second-order Butterworth Filter
# MAGIC ###### REST Reference Method Data was used with the Butterworth Filter

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

######################################## Butterworth Filter ########################################

def butter_bandpass(lowcut, highcut, fs, order=2):
    """
        Design a Butterworth bandpass filter.
        
        Parameters:
        lowcut (float): Lower cutoff frequency
        highcut (float): Upper cutoff frequency
        fs (float): Sampling frequency
        order (int): Order of the filter. Default is 2.
        
        Returns:
        b, a (ndarray, ndarray): Numerator and denominator polynomial coefficients of the filter
    """
    nyq = 0.5 * fs # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
        Applies a Butterworth bandpass filter to a signal.

        Parameters:
        - data: Signal data to be filtered.
        - lowcut: Lower cutoff frequency.
        - highcut: Higher cutoff frequency.
        - fs: Sampling frequency.
        - order: Order of the filter (default is 2).

        Returns:
        - y: Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

######################### Graphs #########################

def plot_butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Plots the frequency response of a Butterworth bandpass filter.

    Parameters:
    - lowcut: Lower cutoff frequency.
    - highcut: Higher cutoff frequency.
    - fs: Sampling frequency.
    - order: Order of the filter (default is 2).
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.figure(figsize=(10, 6))
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), 'b')
    plt.plot(lowcut, 0.5 * np.sqrt(2), 'ko')
    plt.plot(highcut, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(lowcut, color='k')
    plt.axvline(highcut, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Butterworth Bandpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid()
    plt.show()

######################### Constants -- EEG data parameters #########################
#     
# Define frequency bands
frequency_bands = {
    'delta': (2, 4),
    'theta': (4.5, 7.5),
    'alpha': (8, 12.5),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# EEG data parameters
sampling_freq = 250  # frequency in Hz
duration = 900  # Duration of EEG data in seconds
num_samples = sampling_freq * duration
num_channels = 19  # Number of EEG channels

# Define filter parameters
order = 2  # Filter order

######################### REST Reference Method Data Used #########################

# Make a copy of the Dataframe because we are going to apply multiple methods and we dont want to override the original
df_rest_ref_cp = df_rest_ref.copy()

# Get channel names , extract patient_id,index_id and time columns
ch_names = [c for c in df_rest_ref_cp.head() if c != 'patient_id' and c != 'index_id' and c != 'time']

# Extract patient_id column
pt_names = list(df_rest_ref_cp['patient_id'].unique())
print(f"patient_names:::{pt_names}")

butter_filtered_data_rest = {}

for pt in pt_names:
    print("PATIENT_NAME::", pt)
    butter_filtered_data_rest[pt] = {}
    for c_name in ch_names:
        # print("COLUMN_NAME::", c_name)
        if c_name != 'patient_id':
            butter_filtered_data_rest[pt][c_name] = {}
            ch_data = df_rest_ref_cp[c_name].loc[df_rest_ref_cp['patient_id'] == pt]
            for band, (lowcut, highcut) in frequency_bands.items():
                # Get Patients Electrode Readings
                # print("---BAND::", band)
                butter_filtered_data_rest[pt][c_name][band] = butter_bandpass_filter(ch_data.values, lowcut, highcut, sampling_freq, order)

                # Plot the Butterworth bandpass filter frequency response
                plot_butter_bandpass(lowcut, highcut, sampling_freq, order=order)


# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ##### Isolate specific EEG frequencies using a second-order Butterworth Filter
# MAGIC ###### Average Reference Method Data was used with the Butterworth Filter

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

######################################## Butterworth Filter ########################################

def butter_bandpass(lowcut, highcut, fs, order=2):
    """
        Design a Butterworth bandpass filter.
        
        Parameters:
        lowcut (float): Lower cutoff frequency
        highcut (float): Upper cutoff frequency
        fs (float): Sampling frequency
        order (int): Order of the filter. Default is 2.
        
        Returns:
        b, a (ndarray, ndarray): Numerator and denominator polynomial coefficients of the filter
    """
    nyq = 0.5 * fs # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
        Applies a Butterworth bandpass filter to a signal.

        Parameters:
        - data: Signal data to be filtered.
        - lowcut: Lower cutoff frequency.
        - highcut: Higher cutoff frequency.
        - fs: Sampling frequency.
        - order: Order of the filter (default is 2).

        Returns:
        - y: Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

######################### Graphs #########################

def plot_butter_bandpass(freq_range, fs, order=2):
    """
    Plots the frequency response of a Butterworth bandpass filter.

    Parameters:
    - lowcut: Lower cutoff frequency.
    - highcut: Higher cutoff frequency.
    - fs: Sampling frequency.
    - order: Order of the filter (default is 2).
    """
    lowcut, highcut = freq_range
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.figure(figsize=(10, 6))
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), 'b')
    plt.plot(lowcut, 0.5 * np.sqrt(2), 'ko')
    plt.plot(highcut, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(lowcut, color='k')
    plt.axvline(highcut, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Butterworth Bandpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid()
    plt.show()

######################### Constants -- EEG data parameters #########################
#     
# Define frequency bands
frequency_bands = {
    'delta': (2, 4),
    'theta': (4.5, 7.5),
    'alpha': (8, 12.5),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# EEG data parameters
sampling_freq = 250  # frequency in Hz
duration = 900  # Duration of EEG data in seconds
num_samples = sampling_freq * duration
num_channels = 19  # Number of EEG channels

# Define filter parameters
order = 2  # Filter order

######################### Average Reference Method Data Used #########################

# Make a copy of the Dataframe because we are going to apply multiple methods and we dont want to override the original
df_avg_ref_cp = df_avg_ref.copy()

# Get channel names , extract patient_id,index_id and time columns
ch_names = [c for c in df_avg_ref_cp.head() if c != 'patient_id' and c != 'index_id' and c != 'time']

# Extract patient_id column
pt_names = list(df_avg_ref_cp['patient_id'].unique())
# print(f"patient_names:::{pt_names}")

butter_filtered_data_avg = {}


for pt in pt_names:
    print("PATIENT_NAME::", pt)
    butter_filtered_data_avg[pt] = {}
    for c_name in ch_names:
        # print("COLUMN_NAME::", c_name)
        if c_name != 'patient_id':
            butter_filtered_data_avg[pt][c_name] = {}
            ch_data = df_avg_ref_cp[c_name].loc[df_avg_ref_cp['patient_id'] == pt]
            for band, (lowcut, highcut) in frequency_bands.items():
                # Get Patients Electrode Readings
                # print("---BAND::", band)
                butter_filtered_data_avg[pt][c_name][band] = butter_bandpass_filter(ch_data.values, lowcut, highcut, sampling_freq, order)

for freq_range in frequency_bands.values():
    # Plot the Butterworth bandpass filter frequency response
    plot_butter_bandpass(freq_range, sampling_freq, order=order)



# COMMAND ----------

# MAGIC %md
# MAGIC ###### Checking that the order and level of Butterworth Filter DataFrame is correct

# COMMAND ----------

# Testing that the data schema looks correct 

# print(butter_filtered_data.keys())
# for pt in butter_filtered_data:
#     print("PT::", pt)
#     # print("KEYS:",butter_filtered_data[pt].keys())
#     for ch in butter_filtered_data[pt]:
#         print("---CH::", ch)
#         print(" ++ BANDS::", butter_filtered_data[pt][ch].keys())
#         for band, data in butter_filtered_data[pt][ch].items():
#             print(" +++ BAND::", band)
#             print(".  @@@@ DATA::", data);
#             print("     DATA LENGTH::", len(data))

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Creating the Database Table Schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, FloatType

# ELECTRODE_LOCATIONS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3','C3', 'Cz',  'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1','O2']

schema = StructType([
    StructField("patient_id", StringType(), True),
    StructField("time", FloatType(), True),
    StructField("Fp1", DoubleType(), True),
    StructField("Fp2", DoubleType(), True),
    StructField("F7", DoubleType(), True),
    StructField("F3", DoubleType(), True),
    StructField("Fz", DoubleType(), True),
    StructField("F4", DoubleType(), True),
    StructField("F8", DoubleType(), True),
    StructField("T3", DoubleType(), True),
    StructField("C3", DoubleType(), True),
    StructField("Cz", DoubleType(), True),
    StructField("C4", DoubleType(), True),
    StructField("T4", DoubleType(), True),
    StructField("T5", DoubleType(), True),
    StructField("P3", DoubleType(), True),
    StructField("Pz", DoubleType(), True),
    StructField("P4", DoubleType(), True),
    StructField("T6", DoubleType(), True),
    StructField("O1", DoubleType(), True),
    StructField("O2", DoubleType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create PySpark DataFrames for REST Reference Method Butterworth Filter Data 
# MAGIC ###### Create Butterworth Filter data for REST Reference Method Gold Layer Table 

# COMMAND ----------

import pandas as pd
# print(butter_filtered_data)

# Our original dataframe coming from Silver table (df_rest_ref_all) looks like this:
# df_original = pd.DataFrame([
#     {
#         'patient_id':'h07',"time":0,
#     }, {
#         'patient_id':'h07',"time":0.5,
#     },{
#         'patient_id':'h07',"time":0.9,
#     },{
#         'patient_id':'h07',"time":1.2,
#     },
# ])

# Our Butterworth filter dictionary data looks like this:
# butter_filtered_data = {
#     "h07":{
#         "F7":{
#               "delta":[0,1,2,3],
#             "theta":[4,5,6,7]},
#        "T3":{
#            "delta":[9,11,12,13],
#            "theta":[44,135,63,73]}
#         }
# }

######################### REST Reference Method Data Used #########################

# Define frequency bands
freq_bands = {
    'delta',
    'theta',
    'alpha',
    'beta',
    'gamma'
}
# Convert the filtered data dictionary into a DataFrame
rest_band_init = []

for band in freq_bands:
  print("DROP TABLE::", f"main.solution_accelerator.butter_rest_{band}_gold")
  spark.sql(f"DROP TABLE IF EXISTS main.solution_accelerator.butter_rest_{band}_gold");  

# Iterate over the patients in the dictionary
for pt in butter_filtered_data_rest:
  bands_dict_rest = {}
  print(f"PATIENT:::{pt}")
  original_time_points = df_rest_ref['time'].loc[df_rest_ref['patient_id'] == pt].to_list()
  # print('original_time_points:::', original_time_points)
  # Iterate over the channels
  for ch in butter_filtered_data_rest[pt]:
    # Get the number readings from one of the bands for this channel, doesn't matter which band, its assumeed all are the same length
    num_readings = len(butter_filtered_data_rest[pt][ch]['delta'])
   
    # Iterate over the number of readings (assumed to be in time order)
    for indx in range(num_readings):
      # Iterate over each band for the channel (should be the same for each channel)
      for band in butter_filtered_data_rest[pt][ch]:
        # print(f"len band:::{len(butter_filtered_data_rest[pt][ch][band])}")
        # print("PATIENT:::", butter_filtered_data_rest, "TIME::", indx, "BAND::",band, "CHANNEL::", ch, "VALUE::", butter_filtered_data_rest[pt][ch][band][indx])
        # Initialize the bands_dict_rest for this band if it doesn't exist yet
        if band not in bands_dict_rest:
          bands_dict_rest[band] = []
          # Intialize the time point for this band if it doens't exist yet by putting it in the empty dictionary (bands_dict_rest)
        if len(bands_dict_rest[band]) < indx + 1:
          # Add the channel and value to the indx for the band
          bands_dict_rest[band].append({"patient_id": pt, "time": original_time_points[indx]})
        # Add the reading value to the dictionary
        bands_dict_rest[band][indx][ch] = butter_filtered_data_rest[pt][ch][band][indx]
  
  for band in bands_dict_rest:
    df_dict_rest = spark.createDataFrame(bands_dict_rest[band], schema=schema)   
    if band in rest_band_init:
      print("APPEND::", pt, " BAND::", band, f"main.solution_accelerator.butter_rest_{band}_gold")
      df_dict_rest.write.format("delta").mode("append").saveAsTable(f"main.solution_accelerator.butter_rest_{band}_gold")
    else:
      print("OVERWRITE::", pt, " BAND::", band)
      df_dict_rest.write.format("delta").mode("overwrite").saveAsTable(f"main.solution_accelerator.butter_rest_{band}_gold")
      rest_band_init.append(band)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Deleting all DataFrames and dictionaries from memory

# COMMAND ----------

# Delete the Dataframes and dictionary from memory
del butter_filtered_data_rest
del df_dict_rest 
del df_rest_ref 
del df_rest_ref_cp
del bands_dict_rest

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create PySpark DataFrames for Average Reference Method Butterworth Filter Data 
# MAGIC ###### Create Butterworth Filter data for Average Reference Method Gold Layer Table 

# COMMAND ----------

import pandas as pd
# print(butter_filtered_data)

# Our original dataframe coming from Silver table (df_avg_ref_all) looks like this:
# df_original = pd.DataFrame([
#     {
#         'patient_id':'h07',"time":0,
#     }, {
#         'patient_id':'h07',"time":0.5,
#     },{
#         'patient_id':'h07',"time":0.9,
#     },{
#         'patient_id':'h07',"time":1.2,
#     },
# ])

# Our Butterworth filter dictionary data looks like this:
# butter_filtered_data = {
#     "h07":{
#         "F7":{
#             "delta":[0,1,2,3],
#             "theta":[4,5,6,7]},
#        "T3":{
#            "delta":[9,11,12,13],
#            "theta":[44,135,63,73]}
#         }
# }


######################### Average Reference Method Used #########################

# Convert the filtered data dictionary into a DataFrame

# Define frequency bands
freq_bands = {
    'delta',
    'theta',
    'alpha',
    'beta',
    'gamma'
}

avg_band_init = []

for band in freq_bands:
  print("DROP TABLE::", f"main.solution_accelerator.butter_avg_{band}_gold")
  spark.sql(f"DROP TABLE IF EXISTS main.solution_accelerator.butter_avg_{band}_gold");  

# Iterate over the patients in the dictionary
for pt in butter_filtered_data_avg:
  bands_dict_avg = {}
  print(f"PATIENT:::{pt}")
  original_time_points = df_avg_ref['time'].loc[df_avg_ref['patient_id'] == pt].to_list()
  # Iterate over the channels
  for ch in butter_filtered_data_avg[pt]:
    # Get the number readings from one of the bands for this channel, doesn't matter which band, its assumeed all are the same length
    num_readings = len(butter_filtered_data_avg[pt][ch]['delta'])
    # Iterate over the number of readings (assumed to be in time order)
    for indx in range(num_readings):
      # Iterate over each band for the channel (should be the same for each channel)
      for band in butter_filtered_data_avg[pt][ch]:
        # print(f"len band:::{len(butter_filtered_data_avg[pt][ch][band])}")
        # print("TIME::", time_point, "BAND::",band, "CHANNEL::", ch, "VALUE::", butter_filtered_data_avg[pt][ch][band][time_point])
        # Initialize the bands_dict_avg for this band if it doesn't exist yet
        if band not in bands_dict_avg:
          bands_dict_avg[band] = []
          # Intialize the time point for this band if it doens't exist yet by putting it in the empty dictionary (bands_dict_avg)
        if len(bands_dict_avg[band]) < indx + 1:
          # Add the channel and value to the indx for the band
          bands_dict_avg[band].append({"patient_id": pt, "time": original_time_points[indx]})
        
        # Add the reading value to the dictionary
        bands_dict_avg[band][indx][ch] = butter_filtered_data_avg[pt][ch][band][indx]
  for band in bands_dict_avg:
    df_dict_avg = spark.createDataFrame(bands_dict_avg[band], schema=schema)   
    if band in avg_band_init:
      print("APPEND::", pt, " BAND::", band, f"main.solution_accelerator.butter_avg_{band}_gold")
      df_dict_avg.write.format("delta").mode("append").saveAsTable(f"main.solution_accelerator.butter_avg_{band}_gold")
    else:
      print("OVERWRITE::", pt, " BAND::", band)
      df_dict_avg.write.format("delta").mode("overwrite").saveAsTable(f"main.solution_accelerator.butter_avg_{band}_gold")
      avg_band_init.append(band)

# Delete the Dataframes and dictionary from memory
# del butter_filtered_data_avg
# del df_dict_avg
# del df_avg_ref 
# del df_avg_ref_cp
# del bands_dict_avg

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Deleting all DataFrames and dictionaries from memory

# COMMAND ----------

# Delete the Dataframes and dictionary from memory
del butter_filtered_data_avg
del df_dict_avg
del df_avg_ref 
del df_avg_ref_cp
del bands_dict_avg

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
# MAGIC ###### Retrieve data from Silver Table

# COMMAND ----------

# REST Reference Method Table
df_rest_ref_all = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_rest_ref_silver ORDER BY index_id ASC""")
# Average Reference Method Table
df_avg_ref_all = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_avg_ref_silver ORDER BY index_id ASC""")

# Show the result
display(df_rest_ref_all)
# df_rest_ref_all.show()

# Show the result
display(df_avg_ref_all)
# df_avg_ref_all.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Convert PySpark Dataframe to Pandas Dataframe

# COMMAND ----------

# Convert our two PySpark Dataframes to Pandas Dataframes
# display(df_rest_ref_all.head())

df_rest_ref = df_rest_ref_all.toPandas()
display(df_rest_ref)

# Convert our two PySpark Dataframes to Pandas Dataframes
# display(df_avg_ref_all.head())

df_avg_ref = df_avg_ref_all.toPandas()
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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

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
# print(f"patient_names:::{pt_names}")

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


######################### Graphs #########################

# # Print filtered EEG data shape
# for band, data in butter_filtered_data_rest.items():

#     # Compute the frequency response of the filter
#     b, a = butter_bandpass(ch_names, lowcut, highcut, sampling_freq, order)
#     w, h = freqz(b, a, worN=8000)

#     # Convert the normalized frequencies to Hz
#     frequencies = 0.5 * sampling_freq * w / np.pi

#     # Plot the frequency response
#     plt.figure()
#     plt.plot(frequencies, np.abs(h), 'b')
#     plt.title('Butter Filter frequency response for EEG Data each subject.')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Gain')
#     plt.grid()
#     plt.show()

#     plt.semilogx(w, 20 * np.log10(abs(h)))
#     plt.title('Butter Filter frequency response for EEG Data each subject.')
#     plt.xlabel('Frequency [radians / second]')
#     plt.ylabel('Amplitude [dB]')
#     plt.margins(0, 0.1)
#     plt.grid(which='both', axis='both')
#     plt.axvline(100, color='green') # cutoff frequency
#     plt.show()

    # t = np.linspace(0, 1, 1000, False)  # 1 second
    # sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax1.plot(t, sig)
    # ax1.set_title('10 Hz and 20 Hz sinusoids')
    # ax1.axis([0, 1, -2, 2])

    # sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    # filtered = signal.sosfilt(sos, sig)
    # ax2.plot(t, filtered)
    # ax2.set_title('After 15 Hz high-pass filter')
    # ax2.axis([0, 1, -2, 2])
    # ax2.set_xlabel('Time [seconds]')
    # plt.tight_layout()
    # plt.title('Butter Filter frequency response for EEG Data each subject.')
    # plt.show()



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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

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


######################### Graphs #########################

        # # Print filtered EEG data shape
        # for band, data in butter_filtered_data.items():
        #     print(f'{band} band filtered data shape: {data.shape}')

        # # Compute the frequency response of the filter
        # b, a = butter_bandpass(df_patients[c_name].values, lowcut, highcut, sampling_freq, order)
        # w, h = freqz(b, a, worN=8000)

        # # Convert the normalized frequencies to Hz
        # frequencies = 0.5 * sampling_freq * w / np.pi

        # # Plot the frequency response
        # plt.figure()
        # plt.plot(frequencies, np.abs(h), 'b')
        # plt.title('Butter Filter frequency response for EEG Data each subject.')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Gain')
        # plt.grid()
        # plt.show()

        # plt.semilogx(w, 20 * np.log10(abs(h)))
        # plt.title('Butter Filter frequency response for EEG Data each subject.')
        # plt.xlabel('Frequency [radians / second]')
        # plt.ylabel('Amplitude [dB]')
        # plt.margins(0, 0.1)
        # plt.grid(which='both', axis='both')
        # plt.axvline(100, color='green') # cutoff frequency
        # plt.show()

        # t = np.linspace(0, 1, 1000, False)  # 1 second
        # sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # ax1.plot(t, sig)
        # ax1.set_title('10 Hz and 20 Hz sinusoids')
        # ax1.axis([0, 1, -2, 2])

        # sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
        # filtered = signal.sosfilt(sos, sig)
        # ax2.plot(t, filtered)
        # ax2.set_title('After 15 Hz high-pass filter')
        # ax2.axis([0, 1, -2, 2])
        # ax2.set_xlabel('Time [seconds]')
        # plt.tight_layout()
        # plt.title('Butter Filter frequency response for EEG Data each subject.')
        # plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ###### Checking that the order and level of Butterworth Filter DataFrame is correct

# COMMAND ----------

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
# MAGIC %md
# MAGIC ##### Create PySpark DataFrames for REST Reference Method Butterworth Filter Data 

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
#             "delta":[0,1,2,3],
#             "theta":[4,5,6,7]},
#        "T3":{
#            "delta":[9,11,12,13],
#            "theta":[44,135,63,73]}
#         }
# }

# Convert the filtered data dictionary into a DataFrame

bands_dict_rest = {}

######################### REST Reference Method Data Used #########################

# Iterate over the patients in the dictionary
for pt in butter_filtered_data_rest:
  original_time_points = df_rest_ref['time'].loc[df_rest_ref['patient_id'] == pt].to_list()
  # Iterate over the channels
  for ch in butter_filtered_data_rest[pt]:
    # Get the number readings from one of the bands for this channel, doesn't matter which band, its assumeed all are the same length
    num_readings = len(butter_filtered_data_rest[pt][ch]['delta'])
    # Iterate over the number of readings (assumed to be in time order)
    for indx in range(num_readings):
      # Iterate over each band for the channel (should be the same for each channel)
      for band in butter_filtered_data_rest[pt][ch]:
        # print(f"len band:::{len(butter_filtered_data_rest[pt][ch][band])}")
        # print("TIME::", time_point, "BAND::",band, "CHANNEL::", ch, "VALUE::", butter_filtered_data_rest[pt][ch][band][time_point])
        # Initialize the bands_dict_rest for this band if it doesn't exist yet
        if band not in bands_dict_rest:
          bands_dict_rest[band] = []
          # Intialize the time point for this band if it doens't exist yet by putting it in the empty dictionary (bands_dict_rest)
        if len(bands_dict_rest[band]) < indx + 1:
          # Add the channel and value to the indx for the band
          bands_dict_rest[band].append({"patient_id": pt, "time": original_time_points[indx]})
        
        # Add the reading value to the dictionary
        bands_dict_rest[band][indx][ch] = butter_filtered_data_rest[pt][ch][band][indx]


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Gold Layer Table
# MAGIC ###### Butterworth Filter data for REST Reference Method

# COMMAND ----------

# Create Tables
for band in bands_dict_rest.keys():
    # print("LEN::",len(bands_dict_rest[band]))
    # print("FIRST::", bands_dict_rest[band][0])
    # print("LAST::", bands_dict_rest[band][-1])
    df_dict_rest = spark.createDataFrame(bands_dict_rest[band], schema=schema)
    # Dropping the table because we may have updated the Dataframe
    spark.sql(f"DROP TABLE IF EXISTS main.solution_accelerator.butter_rest_{band}_gold");  

    # Establish a persistent delta table by converting the previously created Spark DataFrames into a Delta Tables. Replace any previously existing table and register the DataFrame as a Delta table in the metastore.
    df_dict_rest.write.format("delta").mode("overwrite").saveAsTable(f"main.solution_accelerator.butter_rest_{band}_gold")

    print("Tables exist.")

# Delete the Dataframes and dictionary from memory
del df_dict_rest
del bands_dict_rest

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create PySpark DataFrames for Average Reference Method Butterworth Filter Data 

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

# Convert the filtered data dictionary into a DataFrame

bands_dict_avg = {}

######################### Average Reference Method Used #########################

# Iterate over the patients in the dictionary
for pt in butter_filtered_data_avg:
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
        

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Gold Layer Table
# MAGIC ###### Butterworth Filter data for Average Reference Method

# COMMAND ----------

# Create Tables
for band in bands_dict_avg.keys():
  df_dict_avg = spark.createDataFrame(bands_dict_avg[band], schema=schema)
  # Dropping the table because we may have updated the Dataframe
  spark.sql(f"DROP TABLE IF EXISTS main.solution_accelerator.butter_avg_{band}_gold");  

  # Establish a persistent delta table by converting the previously created Spark DataFrames into a Delta Tables. Replace any previously existing table and register the DataFrame as a Delta table in the metastore.
  df_dict_avg.write.format("delta").mode("overwrite").saveAsTable(f"main.solution_accelerator.butter_avg_{band}_gold")

  print("Tables exist.")

# Delete the Dataframes and dictionary from memory
del df_dict_avg
del bands_dict_avg

# COMMAND ----------

# MAGIC %md
# MAGIC

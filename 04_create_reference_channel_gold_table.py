# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview - Create Reference Channel and Create Gold Table 
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

df_zero_ref_all = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_zero_ref_silver""")
df_avg_ref_all = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_avg_ref_silver""")

# Show the result
display(df_zero_ref_all)
# df_zero_ref_all.show()

# Show the result
display(df_avg_ref_all)
# df_avg_ref_all.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Convert PySpark Dataframe to Pandas Dataframe

# COMMAND ----------

# Convert our two PySpark Dataframes to Pandas Dataframes
display(df_zero_ref_all.head())

df_zero_ref = df_zero_ref_all.toPandas()

# Convert our two PySpark Dataframes to Pandas Dataframes
display(df_avg_ref_all.head())

df_avg_ref = df_avg_ref_all.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Isolate specific EEG frequencies using a second-order Butterworth Filter

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

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

####### Butterworth Filter #######

# For each patient, signals of each EEG channel were filtered

# Make a copy of the Dataframe because we are going to apply two different methods and we dont want to override the original
df_zero_ref_cp = df_zero_ref.copy()

# Extract patient_id column
pt_names = list(df_zero_ref_cp['patient_id'].unique())
print(f"patient_names:::{pt_names}")

filtered_data = {}

for pt in pt_names:
    print("PATIENT_NAME::", pt);
    filtered_data[pt] = {}
    for c_name in df_zero_ref_cp.head():
        print("COLUMN_NAME::", c_name);
        if c_name != 'patient_id':
            filtered_data[pt][c_name] ={}
            ch_data = df_zero_ref_cp[c_name].loc[df_zero_ref_cp['patient_id'] == pt]

            for band, (lowcut, highcut) in frequency_bands.items():
                # Get Patients Electrode Readings

                print("---BAND::", band)
                filtered_data[pt][c_name][band] = butter_bandpass_filter(ch_data.values, lowcut, highcut, sampling_freq, order)
        # else:
        #     # Keep the excluded column as is
        #     filtered_data[c_name] = df_patients[c_name].values 

        # # Print filtered EEG data shape
        # for band, data in filtered_data.items():
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

print(filtered_data.keys())
for pt in filtered_data:
    print("PT::", pt)
    # print("KEYS:",filtered_data[pt].keys())
    for ch in filtered_data[pt]:
        print("---CH::", ch)
        print(" ++ BANDS::", filtered_data[pt][ch].keys())
        for band, data in filtered_data[pt][ch].items():
            print(" +++ BAND::", band)
            print(".  @@@@ DATA::", data);
            print("     DATA LENGTH::", len(data))
# TODO make an array of dataframes from the filtered data dictionary

# Convert the filtered data dictionary into a DataFrame
# df_butter_filtered = pd.DataFrame(filtered_data)

# # Display the first few rows of the filtered DataFrame
# print(df_butter_filtered.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Delete if we run out of time - Finite Impulse Response (FIR) Filters Example

# COMMAND ----------

from scipy.signal import firwin, filtfilt
import numpy as np
import pandas as pd

# TODO: Emergency, it makes sense to me that different patients would have different frequencies that need to be filtered out and this filter should be INDIVIDUAL dataset

# Define the FIR bandpass filter functions
def fir_bandpass(lowcut, highcut, fs, numtaps):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    return taps

def fir_bandpass_filter(data, lowcut, highcut, fs, numtaps):
    taps = fir_bandpass(lowcut, highcut, fs, numtaps)
    y = filtfilt(taps, [1.0], data)
    return y

# Remove columns that are not electrodes
df_channels = df_patients.drop(columns=['patient_id'])

# Filter parameters
lowcut = 1.0   # Lower cutoff frequency in Hz
highcut = 50.0 # Upper cutoff frequency in Hz
fs = 250.0     # Sampling frequency in Hz
numtaps = 101  # Number of filter taps (higher means sharper frequency response)

# Apply the filter to each column in the DataFrame
filtered_data = {}
for column in df_channels.columns:
    filtered_data[column] = fir_bandpass_filter(df_channels[column].values, lowcut, highcut, fs, numtaps)

# Convert the filtered data dictionary to a DataFrame
df_fir_filtered = pd.DataFrame(filtered_data)

# Display the first few rows of the filtered DataFrame
print(df_fir_filtered.head())


# COMMAND ----------

# MAGIC %md
# MAGIC ###### Checking number of rows in DataFrame equals number of rows in original data

# COMMAND ----------

# Output the Dataframe Schema 
df_butter_filtered.printSchema()

# Checking number of rows in Dataframes
rows_all = df_butter_filtered.count()
print(f"Number of N rows {rows_all}")

# COMMAND ----------

# Output the Dataframe Schema 
df_fir_filtered.printSchema()

# Checking number of rows in Dataframes
rows_all = df_fir_filtered.count()
print(f"Number of N rows {rows_all}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Explore dataset based on `summarize` statistics

# COMMAND ----------

dbutils.data.summarize(df_butter_filtered)
dbutils.data.summarize(df_fir_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Examine statistical metrics utilizing Databricks' integrated commands `describe` and `summary` for potential future data manipulation.

# COMMAND ----------

df_butter_filtered.describe()
df_fir_filtered.describe()

# COMMAND ----------

df_butter_filtered.summary()
df_fir_filtered.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Gold Layer Table
# MAGIC ###### Use Butterworth Filter data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dropping the table because we may have updated the Dataframe
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_butter_filtered_gold;

# COMMAND ----------

# Structure and integrate raw data into a Delta table for analysis

# Establish a persistent delta table by converting the previously created Spark DataFrames into a Delta Tables.

# Replace any previously existing table and register the DataFrame as a Delta table in the metastore.
df_butter_filtered.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_butter_filtered_gold")

print("Tables exist.")

# Delete the Dataframes from memory
del df_butter_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC

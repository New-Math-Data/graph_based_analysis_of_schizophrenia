# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview - Data Inspection and Create Silver Table 
# MAGIC
# MAGIC The EEG signals from each channel underwent filtering using a second-order Butterworth Filter across distinct physiological frequency ranges: 2–4 Hz (delta), 4.5–7.5 Hz (theta), 8–12.5 Hz (alpha), 13–30 Hz (beta), and 30–45 Hz (gamma).
# MAGIC
# MAGIC Applying a second-order Butterworth Filter to EEG signals allows you to isolate specific frequency components while reducing the strength or amplitude of others, thereby enabling certain frequency components to pass through relatively unaffected. This process aids in extracting meaningful information from EEG data, thereby enhancing the analysis of brain activity across various frequency bands.
# MAGIC
# MAGIC ##### In this notebook we will:
# MAGIC   * Isolate specific EEG frequencies using a second-order Butterworth Filter
# MAGIC   * Explore dataset based on summarize statistics. 
# MAGIC   * Examine statistical metrics utilizing Databricks' integrated commands `describe` and `summary` for potential future data manipulation.
# MAGIC     * `describe`: Provides statistics including count, mean, standard deviation, minimum, and maximum.
# MAGIC     * `summary`: describe + interquartile range (IQR)
# MAGIC     * Look for missing or null values in the data that will cause outliers or skew the data, zero values are expected with EEG data so no need to wrangle them.
# MAGIC   * Create Silver Layer Table
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Retrieve data from Bronze Table

# COMMAND ----------

df_bronze_control = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_data_bronze_control""")
df_bronze_study = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_data_bronze_study""")

# Union the DataFrames
df_bronze_patients = df_bronze_control.union(df_bronze_study)

# Show the result
df_bronze_patients.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Convert PySpark Dataframe to Pandas Dataframe

# COMMAND ----------

# Convert our two PySpark Dataframes to Pandas Dataframes
display(df_bronze_patients.head())

df_patients = df_bronze_patients.toPandas()

# COMMAND ----------

# # Checking to see the number of rows in filtered Dataframe equal the number of rows in the original unfiltered data
# # Original DF
# columns_names = df_patients.count()
# print(f"original data - number of N rows for columns: {columns_names}")

# # Filtered DF
# columns_names_filtered = df_patients_filtered.count()
# print(f"filtered data - number of N rows for columns: {columns_names_filtered}")

# # Checking that he columns still correctly correlate to the EEG channels 
# df_test = df_bronze_patients.select("Cz")
# array = df_test.collect()

# # Define the number of elements to print
# n = 20

# # Print the first n elements
# print(array[:n])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Isolate specific EEG frequencies using a second-order Butterworth Filter

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt

# TODO: Emergency, it makes sense to me that different patients would have different frequencies that need to be filtered out and this filter should be INDIVIDUAL dataset

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

####### EEG Data all Subjects #######
# EEG Data of both the Control and the Study Subjects

filtered_data = {}
for band, (lowcut, highcut) in frequency_bands.items():
    for c_name in df_patients.columns:
        if c_name != 'patient_id':
            filtered_data[c_name] = butter_bandpass_filter(df_patients[c_name].values, lowcut, highcut, fs, order)
        else:
            # Keep the excluded column as is
            filtered_data[c_name] = df_patients[c_name].values 


# Convert the filtered data dictionary into a DataFrame
df_butter_filtered = pd.DataFrame(filtered_data)

# Display the first few rows of the filtered DataFrame
print(df_butter_filtered.head())
# Print filtered EEG data shape
for band, data in filtered_data.items():
    print(f'{band} band filtered data shape: {data.shape}')

# TODO: Make following graphs to graph out actual data df_butter_filtered
# Compute the frequency response of the filter
b, a = butter_bandpass(lowcut, highcut, sampling_freq, order=2)
w, h = freqz(b, a, worN=8000)

# Convert the normalized frequencies to Hz
frequencies = 0.5 * sampling_freq * w / np.pi

# Plot the frequency response
plt.figure()
plt.plot(frequencies, np.abs(h), 'b')
plt.title('Butterworth bandpass filter frequency response for EEG Data all Subjects using Zero Reference Method')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid()
plt.show()

plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth bandpass filter frequency response for EEG Data all Subjects using Zero Reference Method')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()

t = np.linspace(0, 1, 1000, False)  # 1 second
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('10 Hz and 20 Hz sinusoids')
ax1.axis([0, 1, -2, 2])

sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, sig)
ax2.plot(t, filtered)
ax2.set_title('After 15 Hz high-pass filter')
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.title('Butterworth bandpass filter frequency response for EEG Data all Subjects using Zero Reference Method')
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Finite Impulse Response (FIR) Filters Example

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
# MAGIC ##### Create Silver Layer Table
# MAGIC ###### Use Butterworth Filter data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dropping the table because we may have updated the Dataframe
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_butter_filtered_silver;

# COMMAND ----------

# Structure and integrate raw data into a Delta table for analysis

# Establish a persistent delta table by converting the previously created Spark DataFrames into a Delta Tables.

# Replace any previously existing table and register the DataFrame as a Delta table in the metastore.
df_butter_filtered.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_butter_filtered_silver")

print("Tables exist.")

# Delete the Dataframes from memory
del df_butter_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC

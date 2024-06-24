# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview - Data Inspection and Create Silver Table 
# MAGIC
# MAGIC
# MAGIC
# MAGIC While EEG recordings can be made without a reference electrode, the use of a reference electrode is essential for accurate and meaningful EEG analysis. It helps in grounding the electrical potentials, canceling out common-mode noise, facilitating signal comparison, and enabling various analytical techniques. The choice of reference scheme should be made based on the experimental requirements and analytical considerations.
# MAGIC
# MAGIC The EEG signals from each channel underwent filtering using a second-order Butterworth filter across distinct physiological frequency ranges: 2–4 Hz (delta), 4.5–7.5 Hz (theta), 8–12.5 Hz (alpha), 13–30 Hz (beta), and 30–45 Hz (gamma).
# MAGIC
# MAGIC Applying a second-order Butterworth filter to EEG signals allows you to isolate specific frequency components while reducing the strength or amplitude of others, thereby enabling certain frequency components to pass through relatively unaffected. This process aids in extracting meaningful information from EEG data, thereby enhancing the analysis of brain activity across various frequency bands.
# MAGIC
# MAGIC ##### In this notebook we will:
# MAGIC   * Isolate specific EEG frequencies using a second-order Butterworth filter
# MAGIC   * Contrast the different techniques used in EEG data analysis to re-reference the data.
# MAGIC     * Zero Reference Method
# MAGIC     * Average Reference Method
# MAGIC   * Explore dataset based on summarize statistics. 
# MAGIC   * Examine statistical metrics utilizing Databricks' integrated commands `describe` and `summary` for potential future data manipulation.
# MAGIC     * `describe`: Provides statistics including count, mean, standard deviation, minimum, and maximum.
# MAGIC     * `summary`: describe + interquartile range (IQR)
# MAGIC     * Look for missing or null values in the data that will cause outliers or skew the data, zero values are expected with EEG data so no need to wrangle them.
# MAGIC   * Create Silver Layer Table
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### Note: In this notebook, we will **not** use the previously created Bronze Tables. Although it is good practice to store raw data in a Bronze table for reference, for this notebook, we will modify the EEG data directly from the raw files using Python functions specific to the EEG file type.

# COMMAND ----------

df_bronze_control = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_data_bronze_control""")
df_bronze_study = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_data_bronze_study""")

# Union the DataFrames
df_bronze_patients = df_bronze_control.union(df_bronze_study)

# Show the result
df_bronze_patients.show()

# COMMAND ----------

# Convert our two PySpark Dataframes to Pandas Dataframes
display(df_bronze_patients.head())

df_patients = df_bronze_patients.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Finite Impulse Response (FIR) Filters

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
df_patients_filtered = pd.DataFrame(filtered_data)

# Display the first few rows of the filtered DataFrame
print(df_patients_filtered.head())


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
# MAGIC ##### Isolate specific EEG frequencies using a second-order Butterworth filter

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
# MAGIC ##### Contrast the different techniques used in EEG data analysis to re-reference the data.
# MAGIC
# MAGIC Contrast the EEG reference points generated by employing both the ZR and CAR methods to determine the perferred EEG reference point for our specific use case.
# MAGIC
# MAGIC Both the Zero Reference (ZR) and Common Average Reference (CAR) methods are used in EEG (Electroencephalography) to mitigate common noise sources and spatial biases. However, they differ in their approach to reference point selection and signal processing. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Zero Reference Method
# MAGIC In this method, you choose one electrode to be the reference and subtract its signal from all other electrodes.

# COMMAND ----------

pip install mne

# COMMAND ----------

import os
import mne
import pandas as pd
# import numpy as np
from pyspark.sql.functions import lit

####### CALCULATING THE REFERENCE ELECTRODE USING THE ZERO REFERENCE METHOD #######

# Make a copy of the Dataframe because we are going to apply two different methods and we dont want to override the original
df_butter_filtered_cp = df_butter_filtered.copy()

# TODO: Perform the following steps for each patient
 
# col_names_arr = [c for c in df_butter_filtered.columns]

filtered_data = {}
for col in df_patients.columns:
    if c_name != 'patient_id':
        filtered_data[c_name] = df_patients.values
    else:
        # Keep the excluded column as is
        filtered_data[c_name] = df_patients[c_name].values 

# Get channel values only, extract patient_id column. Numpy will remove the index and column labels
df_numeric = df_butter_filtered_cp.select_dtypes(include=[np.number])

# Convert DataFrame to NumPy array and extract the values and transpose it (channels x samples)
np_samples_array = df_numeric.values.T

# Sampling rate in Hz
sfreq = 250

# Get channel names , extract patient_id column
ch_names = [c for c in df_butter_filtered.columns if c != 'patient_id']
print(f"ref_channels:::{ch_names}")

# Create an info structure needed by MNE
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# Create the MNE Raw object
mne_raw = mne.io.RawArray(filtered_data[col], info)

# Now we have your MNE Raw object and are ready for further analysis

# Specify the reference channels, extract columns 'Fz' and 'Cz'
ref_channels = [c for c in df_butter_filtered_cp.columns if c in ['Fz', 'Cz']]

# Compute the average reference signal from Fz and Cz and apply it
mne_raw.set_eeg_reference(ref_channels=ref_channels)

# Extract EEG data from mne_raw and make a times column
eeg_data, data_time = mne_raw.get_data(return_times=True)
print(eeg_data.shape)
print(data_time.shape)

# Create DataFrame
df_raw_data = pd.DataFrame(data=eeg_data.T, columns=ch_names)

# Add a new column
df_raw_data['data_time'] = data_time


# COMMAND ----------

# Output the Dataframe Schema 
df_zero_ref_subjects_all.printSchema()

# Checking number of rows in Dataframes
rows_all = df_zero_ref_subjects_all.count()
print(f"Number of N rows {rows_all}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Average Reference Method
# MAGIC In this method, you calculate the average signal across all electrodes and subtract this average from each electrode. This method assumes that the average potential of all electrodes represents a good approximation of zero potential.

# COMMAND ----------

# Install required libraries
%pip install pyEDFlib

# COMMAND ----------

import os
import mne
import pandas as pd
import pyedflib
import numpy as np
from pyspark.sql.functions import lit

# Directory path
directory = '/tmp'

# Iterate over files in the directory
for filename in os.listdir(directory):
    print(filename)
    # Path to the EEG file (EDF format file in DBFS)
    if os.path.isfile(os.path.join(directory, filename)):
        if "edf" in filename:
            # Create the file path
            filepath = os.path.join(directory, filename)

            # Read the EDF file
            edf_f = pyedflib.EdfReader(filepath)

            # Extract data from the .edf file

            # Get the number of signals, i.e. number of electrode locations
            n_signals = edf_f.signals_in_file

            # Get the labels of each signal, i.e. electrode location names
            signal_labels = edf_f.getSignalLabels()
            
            print(f"signal_labels:::{signal_labels}")

            # Create a dictionary to hold the data
            signal_data = {}

            # Read each signal into the dictionary
            for i in range(n_signals):
                signal_data[signal_labels[i]] = edf_f.readSignal(i)
                # Verify we have 250 Hz of datapoints, 15 min * 60 sec = 900, 900 * 250 = 225000
                print(f"Num of Signals:::{signal_labels[i]} Num{len( signal_data[signal_labels[i]])}")
            print(signal_data.keys())

            # Close the .edf file
            edf_f._close()
            del edf_f

            # making a Pandas DataFrame first is faster and will ensure the double data type for the signals 
            df_eeg_data = pd.DataFrame.from_dict(signal_data)

            ####### CALCULATING THE REFERENCE ELECTRODE USING THE AVERAGE REFERENCE METHOD #######

            # Calculate the average signal across all channels
            average_reference = np.mean(df_eeg_data, axis=0)

            # Subtract the average reference from each channel
            eeg_data_average_referenced = df_eeg_data - average_reference

            # eeg_data_average_referenced is now re-referenced to the average reference

            fn = filename.replace(".edf","")
            print(f"FN::{fn}")

            df_spark = spark.createDataFrame(eeg_data_average_referenced)
            df_spark = df_spark.withColumn('patient_id', lit(fn))
     
            pt = "h"
            if fn.startswith("s"):
                pt = "s"
            df_avg_ref_subjects_all = df_spark.withColumn('subject', lit(pt))

# COMMAND ----------

# Output the Dataframe Schema 
df_avg_ref_subjects_all.printSchema()

# Checking number of rows in Dataframes
rows_sub_all = df_avg_ref_subjects_all.count()
print(f"Number of N rows {rows_sub_all}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Explore dataset based on `summarize` statistics

# COMMAND ----------

dbutils.data.summarize(df_zero_ref_subjects_all)
dbutils.data.summarize(df_avg_ref_subjects_all)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Examine statistical metrics utilizing Databricks' integrated commands `describe` and `summary` for potential future data manipulation.

# COMMAND ----------

df_zero_ref_subjects_all.describe()
df_avg_ref_subjects_all.describe()

# COMMAND ----------

df_zero_ref_subjects_all.summary()
df_avg_ref_subjects_all.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Silver Layer Table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dropping the table because we may have updated the Dataframe
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_zero_ref_data_silver;
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_avg_ref_data_silver;

# COMMAND ----------

# Structure and integrate raw data into a Delta table for analysis

# Establish a persistent delta table by converting the previously created Spark DataFrames into a Delta Tables.

# Replace any previously existing table and register the DataFrame as a Delta table in the metastore.
df_zero_ref_subjects_all.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_zero_ref_data_silver")
df_avg_ref_subjects_all.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_avg_ref_data_silver")

print("Tables exist.")

# Delete the Dataframes from memory
del df_zero_ref_subjects_all
del df_avg_ref_subjects_all

# COMMAND ----------

# MAGIC %md
# MAGIC

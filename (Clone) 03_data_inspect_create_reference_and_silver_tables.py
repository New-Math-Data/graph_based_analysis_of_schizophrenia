# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview - Data Inspection, Create Reference Channel and Create Silver Tables 
# MAGIC
# MAGIC While EEG recordings can be made without a reference electrode, the use of a reference electrode is essential for accurate and meaningful EEG analysis. It helps in grounding the electrical potentials, canceling out common-mode noise, facilitating signal comparison, and enabling various analytical techniques. The choice of reference scheme should be made based on the experimental requirements and analytical considerations.
# MAGIC
# MAGIC ##### In this notebook we will:
# MAGIC   * Contrast the different techniques used in EEG data analysis to re-reference the data.
# MAGIC     * Reference Electrode Standardization Technique (REST) Method
# MAGIC       - REST is a method used in electrochemistry to ensure that measurements taken with different reference electrodes are comparable.  
# MAGIC     * Average Reference Method
# MAGIC       - In this method, you calculate the average signal across all electrodes and subtract this average from each electrode. This method assumes that the average potential of all electrodes represents a good approximation of zero potential.
# MAGIC     * Zero Reference Method
# MAGIC       - In this method, you choose one electrode to be the reference and subtract its signal from all other electrodes.
# MAGIC   * Explore dataset based on summarize statistics. 
# MAGIC   * Examine statistical metrics utilizing Databricks' integrated commands `describe` and `summary` for potential future data manipulation.
# MAGIC     * `describe`: Provides statistics including count, mean, standard deviation, minimum, and maximum.
# MAGIC     * `summary`: describe + interquartile range (IQR)
# MAGIC     * Look for missing or null values in the data that will cause outliers or skew the data, zero values are expected with EEG data so no need to wrangle them.
# MAGIC   * Create Silver Layer Table

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Retrieve data from Bronze Table

# COMMAND ----------

df_bronze_control = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_data_bronze_control""")
df_bronze_study = spark.sql("""SELECT * FROM main.solution_accelerator.eeg_data_bronze_study""")

# Union the DataFrames
df_bronze_patients = df_bronze_control.union(df_bronze_study)

# Show the result
display(df_bronze_patients)
# df_bronze_patients.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Convert PySpark Dataframe to Pandas Dataframe

# COMMAND ----------

# Convert our two PySpark Dataframes to Pandas Dataframes
display(df_bronze_patients.head())

df_patients = df_bronze_patients.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Contrast the different techniques used in EEG data analysis to re-reference the data.
# MAGIC
# MAGIC Each patient has distinct noise and artifact frequencies that are not part of the usable data and must be identified and filtered out.
# MAGIC
# MAGIC Contrast the EEG reference points generated by employing multiple methods to determine the perferred EEG reference point for our specific data and use case.
# MAGIC
# MAGIC Both the REST and Average Reference (AR) methods are used in EEG (Electroencephalography) to mitigate common noise sources and spatial biases. However, they differ in their approach to reference point selection and signal processing. 
# MAGIC

# COMMAND ----------

# Helper library with many built-in functions
%pip install mne

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Let's make a Databricks method to convert PySpark DataFrames to MNE Objects

# COMMAND ----------

import mne

# Sampling rate in Hz
sfreq = 250

# Get channel names , extract patient_id column
ch_names = [c for c in df_patients.head() if c != 'patient_id']
print(f"ref_channels:::{ch_names}")

# Extract patient_id column
pt_names = list(df_patients['patient_id'].unique())
print(f"patient_names:::{pt_names}")

mne_raw_all = {}

for pt in pt_names:
    print("PATIENT_NAME::", pt)
    # Create an info structure needed by MNE
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    print("LEN::", len(df_patients['patient_id'].index))
    df_pt_data = df_patients.loc[df_patients['patient_id'] == pt]
    df_pt_data = df_pt_data.drop(columns=['patient_id'])
    # Convert Pandas Dataframe to Numpy Array for each patient
    np_pt_data = df_pt_data.to_numpy() 
    # Create the MNE Raw object
    mne_raw = mne.io.RawArray(np_pt_data.T, info)
    mne_raw_all[pt] = mne_raw
    # Plot the data so we can compare graphes later
    mne_raw.plot(duration=100)

    spectrum = mne_raw.compute_psd()
    spectrum.plot(average=True, picks="data", exclude="bads", amplitude=False)

    break
# Now we have our MNE Raw objects and are ready for further analysis


# COMMAND ----------

# MAGIC %md
# MAGIC CSD Stuffs

# COMMAND ----------

import mne

# myraw = mne_raw_all['h12']
# myraw = myraw.set_montage('standard_1020')

# raw_csd = mne.preprocessing.compute_current_source_density(myraw)
# rawspec = myraw.compute_psd()
# rawspec.plot(average=True, picks="data", exclude="bads", amplitude=False)
# csd_spec = raw_csd.compute_psd()
# csd_spec.plot(average=True, picks="data", exclude="bads", amplitude=False)
# hmm = csd_spec.plot_topomap(colorbar=False, res=100)

####### CSD Transformation #######

# Extract patient_id column
pt_names = list(df_patients['patient_id'].unique())
print(f"patient_names:::{pt_names}")

print(f"mne_raw_all{mne_raw_all.keys()}")
display(mne_raw_all)
mne_csd_all = {}
for pt in pt_names:
    mne_raw_pt = mne_raw_all[pt].copy()
    print(f"type mne_raw_pt:::{type(mne_raw_pt)}")
    if isinstance(mne_raw_pt, mne.io.RawArray):
        # Apply CSD Transformation
        mne_raw_pt_w_montage = mne_raw_pt.set_montage('standard_1020')
        csd = mne.preprocessing.compute_current_source_density(mne_raw_pt_w_montage)
        mne_csd_all[pt] = csd

        
        
        



# COMMAND ----------

# MAGIC %md
# MAGIC ###### REST Method
# MAGIC Reference Electrode Standardization Technique infinity reference

# COMMAND ----------

import mne

####### CALCULATING THE REFERENCE ELECTRODE USING THE REST METHOD #######

# Extract patient_id column
pt_names = list(df_patients['patient_id'].unique())
print(f"patient_names:::{pt_names}")

print(f"mne_raw_all{mne_raw_all.keys()}")

mne_rest_all = {}

# Calculate the average signal across all channels for each patient
for pt in pt_names:
    mne_raw_pt = mne_raw_all[pt]
    print(f"type mne_raw_pt:::{type(mne_raw_pt)}")
    if isinstance(mne_raw_pt, mne.io.RawArray):
        # Apply REST Method
        mne_forward = mne_raw_pt.make_forward_solution(mne_raw_pt.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None)

        print(mne_forward)

        mne_rest_all[pt] = mne_raw_pt.copy().set_eeg_reference(ref_channels='REST', forward=mne_forward, projection=False)

        # Plot the average referenced data
        mne_rest_all[pt].plot(n_channels=10, title='REST Method Data', show=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Average Reference Method
# MAGIC This method re-references each electrode to the average potential of all electrodes.

# COMMAND ----------

import mne

####### CALCULATING THE REFERENCE ELECTRODE USING THE AVERAGE REFERENCE METHOD #######

# Extract patient_id column
pt_names = list(df_patients['patient_id'].unique())
print(f"patient_names:::{pt_names}")

print(f"mne_raw_all{mne_raw_all.keys()}")

mne_avg_all = {}

# Calculate the average signal across all channels for each patient
for pt in pt_names:
    mne_raw_pt = mne_raw_all[pt]
    mne_raw_pt.plot(n_channels=19, title='Raw Data', show=True)
    # Apply average reference
    mne_avg_all = mne_raw_pt.copy().set_eeg_reference(ref_channels='average', projection=False)

    # Plot the average referenced data
    mne_avg_all.plot(n_channels=19, title='Average Referenced Data', show=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Zero Reference Method
# MAGIC In this method, you choose one electrode to be the reference and subtract its signal from all other electrodes.

# COMMAND ----------

import os
import mne
import pandas as pd
import numpy as np

####### CALCULATING THE REFERENCE ELECTRODE USING THE ZERO REFERENCE METHOD #######

# # Extract patient_id column
# pt_names = list(df_np['patient_id'].unique())
# print(f"patient_names:::{pt_names}")

# # Perform the following steps for each patient

# filtered_data = {}
# for col in df_np.head():
#     if col != 'patient_id':
#         filtered_data[col] = df_np.values
#     else:
#         # Keep the excluded column as is
#         filtered_data[col] = df_np[col].values 

# # Get channel values only, extract patient_id column. Numpy will remove the index and column labels
# df_numeric = df_np.select_dtypes(include=[np.number])

# # Convert DataFrame to NumPy array and extract the values and transpose it (channels x samples)
# np_samples_array = df_numeric.values.T



# Specify the reference channels, extract columns 'Fz' and 'Cz'
ref_channels = [c for c in df_patients.columns if c in ['Fz', 'Cz']]

# Compute the average reference signal from Fz and Cz and apply it
mne_raw.set_eeg_reference(ref_channels=ref_channels)

# Extract EEG data from mne_raw and make a times column
eeg_data, data_time = mne_raw.get_data(return_times=True)
print(eeg_data.shape)
print(data_time.shape)

# Create DataFrame
df_zero_ref_all = pd.DataFrame(data=eeg_data.T, columns=ch_names)

# Add a new column
df_zero_ref_all['data_time'] = data_time


# COMMAND ----------

# Output the Dataframe Schema 
df_zero_ref_cp.printSchema()

# Checking number of rows in Dataframes
rows_all = df_zero_ref_cp.count()
print(f"Number of N rows {rows_all}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Explore dataset based on `summarize` statistics

# COMMAND ----------

dbutils.data.summarize(df_zero_ref_cp)
dbutils.data.summarize(df_avg_ref_cp)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Examine statistical metrics utilizing Databricks' integrated commands `describe` and `summary` for potential future data manipulation.

# COMMAND ----------

df_zero_ref_cp.describe()
df_avg_ref_cp.describe()

# COMMAND ----------

df_zero_ref_cp.summary()
df_avg_ref_cp.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Silver Layer Table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dropping the table because we may have updated the Dataframe
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_zero_ref_silver;
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_avg_ref_silver;

# COMMAND ----------

# Structure and integrate raw data into a Delta table for analysis

# Establish a persistent delta table by converting the previously created Spark DataFrames into a Delta Tables.

# Replace any previously existing table and register the DataFrame as a Delta table in the metastore.
df_zero_ref_all.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_zero_ref_silver")
df_avg_ref_all.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_avg_ref_silver")

print("Tables exist.")

# Delete the Dataframes from memory
del df_zero_ref_all
del df_avg_ref_all
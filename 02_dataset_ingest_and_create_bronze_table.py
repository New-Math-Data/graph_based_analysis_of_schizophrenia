# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview - Data Ingestion and create Bronze table
# MAGIC
# MAGIC #### Introduction
# MAGIC The Bronze layer is the raw data layer where data is ingested into the data lake in a `Bronze Table` with minimal transformation.
# MAGIC
# MAGIC #### Dataset
# MAGIC Electroencephalography (EEG) data collected from 14 patients with schizophrenia and 14 healthy controls is used in this notebook.
# MAGIC
# MAGIC ##### In this notebook you will:
# MAGIC * Retrieve the EEG dataset.
# MAGIC * Read the EEG data into a DataFrame using the ^*`Delta` format, ensuring ACID transactions and query capabilities.
# MAGIC * Clean the DataFrame by removing invalid characters from the header
# MAGIC * Create the Bronze Layer Delta Table
# MAGIC   * Create a Schema/Database for the table's permanent residence
# MAGIC   * Formulate and Integrate raw data into a Delta table for analysis
# MAGIC
# MAGIC ^*Note: `Delta` refers to Delta Lake, which is an open-source storage layer that brings ACID transactions, scalable metadata handling, and data versioning to processing engines.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Retrieve the dataset from the .zip file stored in Catalog. Data was objected from the Open Data (RepOD) website

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Ingest the data into the Catalog using the Data Ingestion tool
# MAGIC
# MAGIC # Navigate to the Databricks File System (DBFS) to access the zipped dataset and unzip the dataset into a temporary directory
# MAGIC rm -rf /tmp/sz2
# MAGIC mkdir /tmp/sz2
# MAGIC unzip -o /Volumes/main/solution_accelerator/eeg_data_schizophrenia/dataverse_files.zip -d /tmp/sz2
# MAGIC
# MAGIC # Note: The /tmp directory is located within your workspace, not the root tmp directory. This distinction will be addressed when accessing the file.
# MAGIC
# MAGIC # Verify that the file was unzipped by listing the files in the datasets directory
# MAGIC ls /tmp/sz2
# MAGIC
# MAGIC # Display the current working directory, you should see the .edf files
# MAGIC pwd
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The following is an example of writting the raw EEG data to a Delta Bronze Layer table

# COMMAND ----------

# Install required libraries
%pip install mne

# COMMAND ----------

# Create a Schema/Database for the table's permanent residence if not already exists
spark.sql(f"CREATE DATABASE IF NOT EXISTS solution_accelerator LOCATION '/main'")

# Confirm Schema was created
sc = spark.sql(f"SHOW DATABASES")
sc.show()

# The USE command ensures that subsequent code blocks are executed within the appropriate Schema.
spark.sql(f"USE main.solution_accelerator")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dropping the table because we may have updated the Dataframe
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_data_bronze_control;
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_data_bronze_study;

# COMMAND ----------

import mne
import os
import pandas as pd
from pyspark.sql.functions import lit

# Directory path
directory = '/tmp/sz2'

table_h_init = False
table_s_init = False


# Iterate over files in the directory
for filename in os.listdir(directory):
    print(filename)
    # Path to the EEG file (EDF format file in DBFS)
    if os.path.isfile(os.path.join(directory, filename)):
        if "edf" in filename:
            # Create the file path
            filepath = os.path.join(directory, filename)
            mne_raw = mne.io.read_raw_edf(filepath)
            
            # Get the file name
            patient_file_name = filename.replace(".edf","")
            print(f"patient_file_name::{patient_file_name}")

            # Create Pandas DataFrame. Pandas will give us `index`.
            df_mne_pd = mne_raw.to_data_frame(picks=["eeg"])
            
            # MNE gives us `time`, but later we will need to use an index to extrapolate `time`. Add a new index column. We do this so the data goes in, in order, for the sine waves.
            df_mne_pd['index_id'] = df_mne_pd.index
            df_spark = spark.createDataFrame(df_mne_pd)

            df_spark = df_spark.withColumn('patient_id', lit(patient_file_name))
            method = "append"
            if patient_file_name.startswith("h"):
                if not table_h_init:
                    method = "overwrite"
                    table_h_init = True
                print("WRITE PDATA::", patient_file_name, " METHOD::", method)
                df_spark.write.format("delta").mode(method).saveAsTable("main.solution_accelerator.eeg_data_bronze_control")

            if patient_file_name.startswith("s"):
                if not table_s_init:
                    method = "overwrite"
                    table_s_init = True
                print("WRITE PDATA::", patient_file_name, " METHOD::", method)
                df_spark.write.format("delta").mode(method).saveAsTable("main.solution_accelerator.eeg_data_bronze_study")
print('DONE') 

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf /tmp/sz2

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Verify number of rows in DataFrame looks reasonable

# COMMAND ----------

# Verify the table has been created correctly by querying the table
spark.sql("SELECT * FROM main.solution_accelerator.eeg_data_bronze_control LIMIT 3").show()
spark.sql("SELECT * FROM main.solution_accelerator.eeg_data_bronze_study LIMIT 3").show()

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT patient_id, count(*) AS number_of_readings 
# MAGIC FROM main.solution_accelerator.eeg_data_bronze_control
# MAGIC GROUP BY patient_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT patient_id, count(*) AS number_of_readings 
# MAGIC FROM main.solution_accelerator.eeg_data_bronze_study
# MAGIC GROUP BY patient_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Note: there are many more datapoints for the Study patients than the control.
# MAGIC ##### Equation to find the correct number of datapoints: 
# MAGIC
# MAGIC ##### Convert Minutes to Seconds:
# MAGIC     15 minutes (EEG ran for 15mins) × 60 seconds/minute = 900 seconds
# MAGIC
# MAGIC ##### Calculate Total Number of Samples:
# MAGIC     900 seconds × 250 samples/second (Frequency for data was 250 Hz) = 225,000 samples
# MAGIC
# MAGIC ###### Frequency is the rate at which current changes direction per second. It is measured in hertz (Hz), an international unit of measure where 1 hertz is equal to 1 cycle per second. Hertz (Hz) = One hertz is equal to one cycle per second. Cycle = One complete wave of alternating current or voltage.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Alternative Method to ingest EEG Data using the pyedflib library 
# MAGIC The pyedflib method wasn't utilized because subsequent EEG data processing relied on the MNE package, which automatically handles the conversion of voltage units. This ensured that data loaded into Delta Tables, Pandas, and Spark DataFrames consistently maintained units of voltage. Despite testing both methods, they yielded identical results in terms of voltage, without discrepancies in millivolts or microvolts. 

# COMMAND ----------

# Install required libraries
%pip install pyEDFlib

# COMMAND ----------

# Note: you may need to restart the kernel using dbutils.library.restartPython() to use updated packages.
# dbutils.library.restartPython()

# Verify the installation by importing the package
import pyedflib
print("pyedflib version:", pyedflib.__version__)

# COMMAND ----------

import os
import pyedflib
import pandas as pd
from pyspark.sql.functions import lit

##############################  Alternative Method to ingest EEG Data using the pyedflib package  ##############################
"""
The pyedflib method wasn't utilized because subsequent EEG data processing relied on the MNE package, which automatically handles the conversion of voltage units. This ensured that data loaded into Delta Tables, Pandas, and Spark DataFrames consistently maintained units of voltage. Despite testing both methods, they yielded identical results in terms of voltage, without discrepancies in millivolts or microvolts. 
"""
# Directory path
directory = '/tmp'

df_master_h = None
df_master_s = None

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
                # print(f"Num of Signals:::{signal_labels[i]} Num{len( signal_data[signal_labels[i]])}")
            print(signal_data.keys())

            # Close the .edf file
            edf_f._close()
            del edf_f

            # making a Pandas DataFrame first is faster and will ensure the double data type for the signals 
            df_eeg_data = pd.DataFrame.from_dict(signal_data)

            patient_file_name = filename.replace(".edf","")
            print(f"patient_file_name::{patient_file_name}")

            # Add a new index column. We do this so the data goes in, in order for the sine waves
            df_eeg_data['index_id'] = df_eeg_data.index
            # print(df_eeg_data)
            
            df_spark = spark.createDataFrame(df_eeg_data)
            df_spark = df_spark.withColumn('patient_id', lit(patient_file_name))
            # display(df_spark)
            
            if filename.startswith("h"):
                if df_master_h is None:
                    df_master_h = df_spark
                else:
                    df_master_h = df_master_h.union(df_spark) 
                    print("APPEND")
            if filename.startswith("s"):
                if df_master_s is None:
                    df_master_s = df_spark
                else:
                    df_master_s = df_master_s.union(df_spark)
                    print("APPEND")    
print('DONE') 

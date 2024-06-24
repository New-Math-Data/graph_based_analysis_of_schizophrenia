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
# MAGIC * Retrieve the EEG dataset....
# MAGIC * Read the EEG data into a DataFrame using the ^*`Delta` format, ensuring ACID transactions and query capabilities.
# MAGIC * Clean the DataFrame by removing invalid characters from the header
# MAGIC * Create the Bronze Layer Delta Table
# MAGIC   * Create a Schema/Database for the table's permanent residence
# MAGIC   * Formulate and Integrate raw data into a Delta table for analysis
# MAGIC
# MAGIC ^*Note: "Delta" refers to Delta Lake, which is an open-source storage layer that brings ACID transactions, scalable metadata handling, and data versioning to processing engines.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Retrieve the dataset from the .zip file stored in Catalog that was obtained from the Open Data (RepOD) website

# COMMAND ----------

# MAGIC %sh -e
# MAGIC
# MAGIC # Ingest the data into the Catalog using the Data Ingestion tool
# MAGIC
# MAGIC # Navigate to the Databricks File System (DBFS) to access the zipped dataset and unzip the dataset into a temporary directory
# MAGIC unzip -o /Volumes/main/solution_accelerator/eeg_data_schizophrenia/dataverse_files.zip -d /tmp
# MAGIC
# MAGIC # Note: The /tmp directory is located within your workspace, not the root tmp directory. This distinction will be addressed when accessing the file.
# MAGIC
# MAGIC # Verify that the file was unzipped by listing the files in the datasets directory
# MAGIC ls /tmp
# MAGIC
# MAGIC # Display the current working directory, you should see the .edf files
# MAGIC pwd
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The following is an example of writting the raw EEG data to a Delta Bronze Layer table

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
                print(f"Num of Signals:::{signal_labels[i]} Num{len( signal_data[signal_labels[i]])}")
            print(signal_data.keys())

            # Close the .edf file
            edf_f._close()
            del edf_f

            # making a Pandas DataFrame first is faster and will ensure the double data type for the signals 
            df_eeg_data = pd.DataFrame.from_dict(signal_data)

            fn = filename.replace(".edf","")
            print(f"FN::{fn}")

            df_spark = spark.createDataFrame(df_eeg_data)
            df_spark = df_spark.withColumn('patient_id', lit(fn))
            
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

# COMMAND ----------

# Output the Dataframe Schema 
df_master_h.printSchema()
df_master_s.printSchema()

# COMMAND ----------

# Uncomment code to confirm Datatable was created correctly
# display(df_master_s.head(5))
# display(df_master_h.head(5))

# print(df_master_s)
# print(df_master_h)

# COMMAND ----------

# Checking number of rows in Dataframes

rows_h = df_master_h.count()
print(f"Number of N rows {rows_h}")

rows_s = df_master_s.count()
print(f"Number of Z rows {rows_s}")

# COMMAND ----------

# Create a Schema/Database for the table's permanent residence if not already exists
spark.sql(f"CREATE DATABASE IF NOT EXISTS solution_accelerator LOCATION '/main'")

# Confirm Schema was created
df = spark.sql(f"SHOW DATABASES")
df.show()

# The USE command ensures that subsequent code blocks are executed within the appropriate Schema.
spark.sql(f"USE main.solution_accelerator")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Dropping the table because we may have updated the Dataframe
# MAGIC
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_data_bronze_control;
# MAGIC DROP TABLE IF EXISTS main.solution_accelerator.eeg_data_bronze_study;

# COMMAND ----------

# Structure and integrate raw data into a Delta table for analysis

# Establish a persistent delta table (Bronze Layer table - raw data) by converting the previously created Spark DataFrame into a Delta Table.

# Replace any previously existing table and register the DataFrame as a Delta table in the metastore.
df_master_h.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_data_bronze_control")
df_master_s.write.format("delta").mode("overwrite").saveAsTable("main.solution_accelerator.eeg_data_bronze_study")

print("Table `eeg_data_bronze_control` and `eeg_data_bronze_study` exists.")

# Delete the Dateframe from memory
del df_master_h
del df_master_s

# COMMAND ----------

# Verify the table has been created correctly by querying the table
spark.sql("SELECT * FROM main.solution_accelerator.eeg_data_bronze_control LIMIT 3").show()
spark.sql("SELECT * FROM main.solution_accelerator.eeg_data_bronze_study LIMIT 3").show()

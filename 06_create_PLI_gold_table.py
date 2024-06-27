# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview - Graphing brain connectivity in schizophrenia from EEG data - Create PLI Graphs and Gold Table
# MAGIC
# MAGIC EEG analysis was carried out using:
# MAGIC 1. the raw EEG data, 
# MAGIC as well as the re-referenced data: 
# MAGIC 2. the Average Reference Method and
# MAGIC 3. the Zero Reference Method.
# MAGIC This allowed us to explore how the choice of reference electrode impacts connectivity outcomes.
# MAGIC
# MAGIC EEG data were analyzed using three connectivity methods: Phase-Locking Value (PLV), Phase-Lag Index (PLI), and Directed Transfer Function (DTF), and statistical indices based on graph theory. 
# MAGIC
# MAGIC ##### In this notebook we will:
# MAGIC   * Graph analysis of EEG data measuring connectivity using three connectivity measures:
# MAGIC     * Directed Transfer Function (DTF)
# MAGIC     * Phase-Locking Value (PLV)
# MAGIC     * Phase-Lag Index (PLI)
# MAGIC ##### This Notebook will use Phase-Locking Value (PLV)

# COMMAND ----------

# Load the data from the Butterworth Filtered Data REST Tables
#pt_to_display = ['s11', 'h11'] if the gold tables only include patients of interest no need to select here
band_names = ["delta","theta", "alpha", "beta", "gamma"]

df_bands_rest = {}
# Create Pandas DataFrames
for band in band_names:
    df_bands_rest[band] = spark.sql(f"SELECT * FROM main.solution_accelerator.butter_rest_{band}_gold ORDER BY time ASC")

display(df_bands_rest[band])

# COMMAND ----------

from scipy import signal
import numpy as np
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
# define a couple functions we will need
def instantaneousPhase(sig):
    h = signal.hilbert(sig)
    return np.angle(h)

def pli(sig1, sig2):
    instPhase1 = instantaneousPhase(sig1)
    instPhase2 = instantaneousPhase(sig2)
    phaseDiff = instPhase1 - instPhase2
    phaseDiff = (phaseDiff + np.pi) % (2*np.pi) - np.pi
    pli = abs(np.mean(np.sign(phaseDiff)))
    return pli

def adj_matrix(pd_df):
    channels = pd_df.columns[1:20]
    adjMatrix = np.zeros((len(channels), len(channels)))

    for i in range(len(channels)):
        for j in range(len(channels)):
            sig1 = pd_df[channels[i]].values
            sig2 = pd_df[channels[j]].values
            thispli = pli(sig1, sig2)
            adjMatrix[i, j] = thispli

    return adjMatrix

my_udf = udf(instantaneousPhase, FloatType())

df_pli = df_bands_rest['alpha'].withColumn("c1c2", my_udf("Fp2"))

display(df_pli)


# for band in band_names:
#     df = df_bands_rest[band]
#     unique_patient_ids = df.select("patient_id").distinct().toPandas()['patient_id'].values
#     for ptid in unique_patient_ids:
#         display(ptid)
#         pd_df = df.filter(df.patient_id == ptid).toPandas()
#         adjMatrix = adj_matrix(pd_df)
#         display(adjMatrix)






# COMMAND ----------

results_df


# COMMAND ----------

!pip install seaborn

# COMMAND ----------

import seaborn as sns
sns.heatmap(heatmap, cmap='Spectral_r', 
            square=True, vmin=0.0, linewidths=.5, cbar_kws={"shrink": .5})

# COMMAND ----------

!pip install pyvis

# COMMAND ----------



# COMMAND ----------



# Databricks notebook source
df_silver = spark.sql(""" SELECT * FROM main.solution_accelerator.eeg_zero_ref_data_silver""")

display(df_silver)


# COMMAND ----------

!pip install scot

# COMMAND ----------

!pip install networkx

# COMMAND ----------

#from https://github.com/ufvceiec/EEGRAPH/blob/481d3fe60115a1c6141c4e466144b08b609bfa6c/eegraph/strategy.py#L239

import numpy as np
from tools import * 

# COMMAND ----------

# from eegraph
# which depends on tools.py from eegraph/tools.py
def calculate_conn(data_intervals, i, j, sample_rate, channels, bands):
        sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma = calculate_bands_fft(data_intervals[i], sample_rate, bands)
        sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma = calculate_bands_fft(data_intervals[j], sample_rate, bands)
        
        sig1_bands = instantaneous_phase([sig1_delta, sig1_theta, sig1_alpha, sig1_beta, sig1_gamma])
        sig2_bands = instantaneous_phase([sig2_delta, sig2_theta, sig2_alpha, sig2_beta, sig2_gamma])
        
        complex_phase_diff_delta = np.exp(complex(0,1)*(sig1_bands[0] - sig2_bands[0]))
        complex_phase_diff_theta = np.exp(complex(0,1)*(sig1_bands[1] - sig2_bands[1]))
        complex_phase_diff_alpha = np.exp(complex(0,1)*(sig1_bands[2] - sig2_bands[2]))
        complex_phase_diff_beta = np.exp(complex(0,1)*(sig1_bands[3] - sig2_bands[3]))
        complex_phase_diff_gamma = np.exp(complex(0,1)*(sig1_bands[4] - sig2_bands[4]))
        
        plv_delta = np.abs(np.sum(complex_phase_diff_delta))/len(sig1_bands[0])
        plv_theta = np.abs(np.sum(complex_phase_diff_theta))/len(sig1_bands[1])
        plv_alpha = np.abs(np.sum(complex_phase_diff_alpha))/len(sig1_bands[2])
        plv_beta = np.abs(np.sum(complex_phase_diff_beta))/len(sig1_bands[3])
        plv_gamma = np.abs(np.sum(complex_phase_diff_gamma))/len(sig1_bands[4])
        
        return plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma

# COMMAND ----------

# MAGIC %md
# MAGIC Testing to see if having tools.py works and if calling def calculate_conn works.

# COMMAND ----------

#testing the PLV def
data_intervals = [np.array([1, 2, 3]), np.array([4, 5, 6])]
i = 0
j = 1
sample_rate = 100
channels = 2
bands = 5

plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma = calculate_conn(data_intervals, i, j, sample_rate, channels, bands)

# COMMAND ----------

print(plv_delta)

# COMMAND ----------

df_patient13 = df_silver[df_silver.patient_id == 's13']
display(df_patient13)

# COMMAND ----------

display(df_silver)

# COMMAND ----------



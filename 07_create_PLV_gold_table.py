# Databricks notebook source
df_silver = spark.sql(""" SELECT * FROM main.solution_accelerator.butter_avg_delta_gold where patient_id == 's11'""")

display(df_silver)


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC next few cells for eegraph

# COMMAND ----------

!pip install scot
!pip install networkx

# COMMAND ----------

import numpy as np
from tools import * 

# COMMAND ----------

# MAGIC %md 
# MAGIC unsure about memory impacts of using grouping.  decided to use FOR loop thinking that it will minimize performance problems? 
# MAGIC

# COMMAND ----------

#from https://github.com/ufvceiec/EEGRAPH/blob/481d3fe60115a1c6141c4e466144b08b609bfa6c/eegraph/strategy.py#L239

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

from scipy.signal import hilbert 
import numpy as np
import pandas as pd

# COMMAND ----------

ELECTRODE_LOCATIONS = ['Fp2', 'F8', 'T4', 'T6', 'O2', 'Fp1', 'F7', 'T3', 'T5', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz']

ELECTRODE_LOCATIONS_ORDERED = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3','C3', 'Cz',  'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1','O2']

items = ELECTRODE_LOCATIONS_ORDERED
pairs_with_self = []
for i in range(len(items)):
    for j in range(i, len(items)):
        pairs_with_self.append((items[i], items[j]))
display(pairs_with_self)

# COMMAND ----------

# setup for PLV 

#for EEGraph
i = 0
j = 1
sample_rate = 250
channels = 2
bands = 5

# removed 'subject' since it isn't present in the table,
results_df = pd.DataFrame(columns=['patient_id', 'cx', 'cy', 'plv_delta', 'plv_theta', 'plv_alpha', 'plv_beta', 'plv_gamma'])

unique_patient_ids = df_silver.select("patient_id").distinct()
unique_patient_ids_list = unique_patient_ids.collect()
display(unique_patient_ids_list)

# COMMAND ----------

# if using eegraph
for row in unique_patient_ids_list:
    patient_id = row['patient_id']
    display("starting patient")
    display(patient_id)
    df = df_silver[df_silver.patient_id == patient_id]
    df_pandas = df.toPandas()

    for p in pairs_with_self:
        #display(p[0], p[1])
        data_intervals = [df_pandas[p[0]].head(1000), df_pandas[p[1]].head(1000)]
        plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma = calculate_conn(data_intervals, i, j, sample_rate, channels, bands)
        new_row = {
            'patient_id': df_pandas['patient_id'].iloc[0],
            #'subject': df_patient13_pandas['subject'],
            'cx': p[0],
            'cy': p[1],
            'plv_delta': plv_delta,
            'plv_theta': plv_theta,
            'plv_alpha': plv_alpha,
            'plv_beta': plv_beta,
            'plv_gamma': plv_gamma
        }
        results_df = results_df.append(new_row, ignore_index=True)

# COMMAND ----------

#trying scipy.signal 
for row in unique_patient_ids_list:
    patient_id = row['patient_id']
    display("starting patient")
    display(patient_id)
    df = df_silver[df_silver.patient_id == patient_id]
    df_pandas = df.toPandas()

    for p in pairs_with_self:
        if (p[0] != 'Fp1' and p[1] != 'Fp1'):
            display(p[0], p[1])
            data_intervals = [df_pandas[p[0]], df_pandas[p[1]]]
            
            # Step 1: Compute the Hilbert transform to get the analytic signal
            analytic_signal = np.array([hilbert(channel) for channel in data_intervals])
            #display(plv_delta)
    
            # Step 2: Extract the instantaneous phase
            instantaneous_phase = np.angle(analytic_signal)
            
            # Step 3: Calculate the PLV
            #n_channels = data_intervals.shape[0] FIX THIS IF NEEDING FLEXIBILITY
            n_channels = 2
            plv_matrix = np.zeros((n_channels, n_channels))
            
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    # Calculate the phase difference
                    phase_diff = instantaneous_phase[i] - instantaneous_phase[j]
                    # Calculate the PLV
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_matrix[i, j] = plv
                    plv_matrix[j, i] = plv  # PLV is symmetric
            
            #return plv_matrix
            display(plv_matrix)
            plv_delta = plv_matrix[0, 1]

            new_row = {
                'patient_id': df_pandas['patient_id'].iloc[0],
                #'subject': df_patient13_pandas['subject'],
                'cx': p[0],
                'cy': p[1],
                'plv_delta': plv_delta,
                'plv_theta': 0,
                'plv_alpha': 0,
                'plv_beta': 0,
                'plv_gamma': 0
            }
            results_df = results_df.append(new_row, ignore_index=True)
       

# COMMAND ----------

results_df

# COMMAND ----------

spark_df = spark.createDataFrame(results_df[results_df.patient_id == 's13'])
display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Finding that the Databricks heatmap cannot be customized as needed to match the research paper figures. 

# COMMAND ----------

df_delta = spark_df.select('cx', 'cy', 'plv_delta')
display(df_delta)

# COMMAND ----------

!pip install seaborn

# COMMAND ----------

import seaborn as sns

# COMMAND ----------

a = np.array(df_delta)
display(a)

# COMMAND ----------

filtered_df = results_df[results_df.patient_id == 's11']

# tried pivot, but it was sorting ABC and doesn't have a param for sort
df_pivot = filtered_df.pivot("cx", "cy", "plv_delta")
display(df_pivot)

# COMMAND ----------



#will need to decide about the aggfunc, how to handle that
pivot_table = pd.pivot_table(filtered_df, values='plv_delta', index='cx', columns='cy', aggfunc='mean', sort=False)
display(pivot_table)


# COMMAND ----------

transposed_pivot_table = pivot_table.transpose()
combined_pivot_table = pivot_table.combine_first(transposed_pivot_table)
display(combined_pivot_table)

# COMMAND ----------

sns.heatmap(df_pivot, cmap='Spectral_r', 
            square=True, vmin=0.0, linewidths=.5, cbar_kws={"shrink": .5})

# COMMAND ----------


#vmin=0.0, vmax=1.0, center=0,
sns.heatmap(combined_pivot_table, cmap='Spectral_r', 
            square=True, vmin=0.0, linewidths=.5, cbar_kws={"shrink": .5})

# COMMAND ----------

# MAGIC %md
# MAGIC following section is scratch.  can probably be deleted during final code cleanup

# COMMAND ----------

df_patient13 = df_silver[df_silver.patient_id == 's13']
display(df_patient13)

# COMMAND ----------

df_pair = df_patient13['patient_id', 'Fp1', 'Fp2']
display(df_pair)

# COMMAND ----------

#converting to pandas DF to get np arrays
df_p = df_pair.toPandas()
display(df_p)

# COMMAND ----------

# MAGIC %md
# MAGIC trying to understand correct parameters - using entire column results in exception
# MAGIC
# MAGIC later cell runs by using a subset of the values

# COMMAND ----------


data_intervals = [df_p['Fp1'], df_p['Fp2']]

i = 0
j = 1
sample_rate = 250
channels = 2
bands = 5

plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma = calculate_conn(data_intervals, i, j, sample_rate, channels, bands)

display(plv_delta)
display(plv_theta)
display(plv_alpha)
display(plv_beta)
display(plv_gamma)

# COMMAND ----------

# verifying there are no NAs
display(df_p.Fp1[df_p.Fp1.isna() == True])
display(df_p.Fp2[df_p.Fp2.isna() == True])

# COMMAND ----------

# verifying lengths are the same
print(len(df_p['Fp1']))
print(len(df_p['Fp2']))


# COMMAND ----------


#moving ahead by taking subset of the values, to get past exception
data_intervals = [df_p['Fp1'].head(1000), df_p['Fp2'].head(1000)]
i = 0
j = 1
sample_rate = 250
channels = 2
bands = 5

plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma = calculate_conn(data_intervals, i, j, sample_rate, channels, bands)

display(plv_delta)
display(plv_theta)
display(plv_alpha)
display(plv_beta)
display(plv_gamma)

# COMMAND ----------

# MAGIC %md
# MAGIC now that one is calcualating, calculate all combinations per patient

# COMMAND ----------

#get electrode locations
df_patient13_pandas = df_patient13.toPandas() 
all_cols = list(df_patient13_pandas)
display(all_cols)

# COMMAND ----------



i = 0
j = 1
sample_rate = 250
channels = 2
bands = 5

plv_delta, plv_theta, plv_alpha, plv_beta, plv_gamma = calculate_conn(data_intervals, i, j, sample_rate, channels, bands)


for idx, e in enumerate(ELECTRODE_LOCATIONS):
    display(idx)
    #display(df_patient13_pandas[e])

# COMMAND ----------

for p in pairs_with_self:
    display(p[0], p[1])

# COMMAND ----------

display(results_df['patient_id'].unique())

# COMMAND ----------

# MAGIC %md
# MAGIC cells below are scratch.  can probably be deleted during final code cleanup 

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

print(plv_delta)

# COMMAND ----------



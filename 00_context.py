# Databricks notebook source
# MAGIC %md
# MAGIC You can find this series of notebooks at https://github.com/New-Math-Data/graph_based_analysis_of_schizophrenia

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center">
# MAGIC <img src="https://wnu.7a7.myftpupload.com/wp-content/uploads/2022/03/newmathdatalogo.png" width="400" alt="New Math Data Logo">
# MAGIC </div>
# MAGIC <br>
# MAGIC <br>
# MAGIC
# MAGIC ## Databricks Solution Accelerator Graph-based Analysis of Schizophrenia
# MAGIC
# MAGIC
# MAGIC ### About this series of Notebooks
# MAGIC [INSERT into]
# MAGIC
# MAGIC The goal of the Graph-based Analysis of Schizophrenia Solution Accelerator is to provide a graph-based prediction tool for [INSERT]....
# MAGIC
# MAGIC #### Overview
# MAGIC * This series of notebooks is intended to help research and medical staff [INSERT]....
# MAGIC
# MAGIC In support of this goal, we will:
# MAGIC
# MAGIC * Clean and filter the EEG data provided [INSERT]....
# MAGIC
# MAGIC * Design and build graphical tools for analyzing EEG datasets.
# MAGIC
# MAGIC * Design and develop a machine learning based prediction model.
# MAGIC
# MAGIC * Examine the prediction accuracy of the proposed model using statistical based calculations, the performance assessment metric Root Mean-Square Error (RMSE), and the Coefficient of determination (R2) are used to compare the performance of the model.
# MAGIC
# MAGIC ##### Data collection - publicly available dataset
# MAGIC
# MAGIC * The EEG Data is comprised of 14 patients (7 males: 27.9 ± 3.3 years, 7 females: 28.3 ± 4.1 years) with paranoid schizophrenia, who were hospitalized at the Institute of Psychiatry and Neurology in Warsaw, Poland.
# MAGIC
# MAGIC     Fifteen minutes of EEG data was recorded in all subjects during an eyes-closed resting state condition. The control group was matched in gender and age to the 14 patients completing the study.
# MAGIC     
# MAGIC     Data were acquired with the sampling frequency of 250 Hz using the standard 10–20 EEG montage with 19 EEG channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2.
# MAGIC
# MAGIC     Data located in a public repository: Olejarczyk, E.; Jernajczyk, W. (2017) EEG in schizophrenia. RepOD. http://dx.doi.org/10.18150/repod.0107441.
# MAGIC
# MAGIC
# MAGIC ##### Assumptions
# MAGIC * The EEG brain connectivity data, of the control subjects and the Schizophrenia (Study) subjects, had no other brain related diagnoses, e.g. Epilepsy, Multiple Sclerosis.
# MAGIC
# MAGIC ##### Findings
# MAGIC * None
# MAGIC
# MAGIC **Authors**
# MAGIC - Ramona Niederhausern  [<rnieder@newmathdata.com>]
# MAGIC - Ben Martin [<bmartin@newmathdata.com>]
# MAGIC - Heather Woods [<hwoods@newmathdata.com>]
# MAGIC - Ryan Johnson [<rjohnson@newmathdata.com>]
# MAGIC - Traey Hatch [<thatch@newmathdata.com>]
# MAGIC
# MAGIC **Contact** \
# MAGIC     New Math Data, LLC \
# MAGIC     Phone: +1 (281)  817 - 6190 \
# MAGIC     Email: info@newmathdata.com \
# MAGIC     NewMathData.com
# MAGIC
# MAGIC ___

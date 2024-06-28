# Databricks notebook source
# MAGIC %md
# MAGIC You can find this series of notebooks at https://github.com/New-Math-Data/graph_based_analysis_of_schizophrenia

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overview - Setup
# MAGIC ##### In this notebook you will:
# MAGIC * Configure Databricks Repo GIT Environment
# MAGIC * Configure the Solution Accelerator Environment
# MAGIC * Check, and optionally update, the Databricks CLI version
# MAGIC * Learn about the Databricks and the PySpark Session

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Configure Databricks Repo GIT Environment
# MAGIC Set up git repo configuration using the tutorial provided here: https://partner-academy.databricks.com/learn/course/1266/play/7844/integrating-with-databricks-repos

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Configure the Solution Accelerator Environment
# MAGIC All python packages needed for this Solution Accelerator are pre-installed in the Databricks environment

# COMMAND ----------

# MAGIC %sh 
# MAGIC # Check the version of your default installation of the CLI
# MAGIC databricks -v

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks CLI versions 0.18 and below is the “legacy” CLI. If your version is legacy, we need to update the Databricks CLI

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Intall required packages

# COMMAND ----------

# Install required libraries
%pip install pyEDFlib

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Troubleshooting Import Errors
# MAGIC
# MAGIC 1. Click on the carrot (down arrow) next to your compute cluster and select `Restart`.
# MAGIC 2. If the error persists, terminate the cluster and then start it again.
# MAGIC 3. Run this Notebook in future Notebooks: `%run "./01_setup"`

# COMMAND ----------

# Note: you may need to restart the kernel using dbutils.library.restartPython() to use updated packages.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### About Databricks and the PySpark Session
# MAGIC
# MAGIC PySpark is pre-installed in Databricks notebook.
# MAGIC
# MAGIC The **`SparkSession`** class is the single entry point to all functionality in Spark using the DataFrame API.
# MAGIC
# MAGIC In Databricks notebooks, the SparkSession is created for you (spark = SparkSession.builder.getOrCreate()), and stored in the variable `spark`.
# MAGIC
# MAGIC In this Databricks Wind Turbine Load Prediction Solution Accelerator notebook, the `spark` object is used to create DataFrames, register DataFrames as tables and execute SQL queries.
# MAGIC

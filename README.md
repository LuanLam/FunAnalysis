# FunAnalysis

Use SCP to copy jar files to a folder in the cluster home. All input and output are put in the hdfs using command lines

The following command is used to submit the spark job to Amazon EMR (using "Add Step" procedure): --num-executors 6 --executor-cores 6 --executor-memory 3g --jars /home/hadoop/externalLib/commons-csv-1.1.jar,/home/hadoop/externalLib/univocity-parsers-1.5.1.jar,/home/hadoop/externalLib/spark-csv_2.10-1.2.0.jar



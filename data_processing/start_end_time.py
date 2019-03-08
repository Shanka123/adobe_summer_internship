import findspark
findspark.init()
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

def extract_start_time(spark, files, output):    
    t = []
    for day_file in files:
        df_day = spark.read.csv(day_file, header=True)
        df_day1 = df_day.withColumn("mesos_node_timestamp", df_day['mesos_node_timestamp'].cast(FloatType()))
        df_min = df_day1.filter(df_day1.mesos_node_timestamp > 0).groupby('mesos_task_id').min('mesos_node_timestamp').collect()
        for x in df_min:
            t.append((x['mesos_task_id'], x['min(mesos_node_timestamp)']))

    df = pd.DataFrame(t, columns = ['mesos_task_id', 'start_time'])

    df = df.groupby('mesos_task_id', as_index= False)['start_time'].min()

    df.to_csv(output, index=False)

def extract_end_time(spark, files, output):  
    t = []
    for day_file in files:
        df_day = spark.read.csv(day_file, header=True)
        df_day1 = df_day.withColumn("mesos_node_timestamp", df_day['mesos_node_timestamp'].cast(FloatType()))
        df_min = df_day1.filter(df_day1.mesos_node_timestamp > 0).groupby('mesos_task_id').max('mesos_node_timestamp').collect()
        for x in df_min:
            t.append((x['mesos_task_id'], x['max(mesos_node_timestamp)']))

    df = pd.DataFrame(t, columns = ['mesos_task_id', 'start_time'])

    df = df.groupby('mesos_task_id', as_index= False)['start_time'].min()

    df.to_csv(output, index=False)

if '__name__' == '__main__':
    spark = SparkSession.builder \
            .master("local") \
            .appName("Session") \
            .config(conf=SparkConf()) \
            .getOrCreate()
    files = ['27_Mreduced', '28_Mreduced']
    output = 'start_time_27_to_28.csv'
    extract_start_time(spark, files, output)

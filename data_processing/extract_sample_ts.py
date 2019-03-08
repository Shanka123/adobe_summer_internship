import findspark
findspark.init()
from pyspark.conf import SparkConf
from pyspark import SparkContext
import csv
from pyspark.sql import SparkSession
import pandas as pd

def find_next_sample_timestamp(arr, x, s):
    for i in range(s+1, len(arr)-1):
        if(arr[i] <= x and x <= arr[i+1]):
            if(abs(arr[i] - x) < (arr[i+1] - x)):
                return i
            else:
                return i+1
    return -1
"""
For all the given tasks, it will genearate separate csv which contain the utilization of the tasks and also does the smapling 
input:
  files: files corresponding for each day
  tasks: task ids for which timeseries to be extracted 
  output_dir: the directory where ouput csv will be stored
  sample_period: sampling duration in seconds
  start_timestamp: timestamp from where data points needs to be collected
"""
def extract_sample_ts(files, tasks, output_dir,sample_period = 300, start_timestamp = 1527378900):
    def timestamp(a, b):
        return (a[0] + b[0], a[1] + b[1])

    timestamps = [(start_timestamp + i*sample_period) for i in range((len(files)*24*60*60) // sample_period)]
    
    csv_file = sc.textFile(",".join(files), 60)
    csv_file2 = csv_file.map(lambda line: line.split(','))\
                .filter(lambda line: line[0] in tasks and len(line) > 5)\
                .map(lambda word: ((word[0], float(word[4])), ( float(word[1]) , 1) ))\
                .reduceByKey(timestamp)
    
    x1 = csv_file2.collect()

    y = [(x, y[0]/y[1]) for (x,y) in x1]
    y.sort()
    z = {}
    for elem in y:
        if elem[0][0] not in z.keys():
            z[elem[0][0]] = []
        if float(elem[0][1]) > 0:
            z[elem[0][0]].append((elem[0][1], elem[1]))

    for key, x in z.items():
        filename = output_dir+key+ '.csv'
        sampled_ts = []
        i=0
        available_timestamps = [row[0] for row in x]
        j = find_next_sample_timestamp(timestamps, available_timestamps[0], -1)
        k = 0
        for i in range(j+1, len(timestamps)):
            k = find_next_sample_timestamp(available_timestamps, timestamps[i], k)
            if(k==-1):
                break
            sampled_ts.append((timestamps[i], x[k][1]))

        dfn = pd.DataFrame(sampled_ts, columns = ['node_timestamp', 'cpu_time'])
        dfn['utilization'] = (dfn['cpu_time'].diff() / dfn['node_timestamp'].diff()) * 100
        dfn = dfn.drop(columns = ['cpu_time'])
        dfn = dfn[1:]
        dfn.to_csv(filename, index=False)

if '__name__' == '__main__':
    files = ['27.csv', '28.csv']
    tasks = ['activitylogreceiver-prod-0bacf0ebf9.activitylogreceiver---51040c45e1a519ee176c8daa9e77ba6378a944e0----72b845.6b9a88cd-5e7e-11e8-9273-2e60ad6fcb96','scidxsvc-production-36a1ff80c9.scidx---44f76356a831cc5b857e35363ca55b45bcb98a9c----44ed60.edbf821c-5fff-11e8-bcd1-8633960b35b2',
      'tps-prod-2d4e5b3fe2.tps---66893b06de7a86adcc1972ffa5bad5168ca15a6c----332e03.ec12717f-6028-11e8-a239-421db0942d0d','lridxsvc-production-94333b4209.lridx---207a8ec7f8378dd28d301c08e511a910e04da7ce----18c0d6.39f62255-5fa3-11e8-a239-421db0942d0d']
    extract_sample_ts(files, tasks, '/tmp/',sample_period = 300, start_timestamp = 1527378900):

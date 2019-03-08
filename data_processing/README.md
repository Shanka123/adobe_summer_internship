1. [pre_process_time_series.py](./pre_process_time_series.py)
  
    It convert the values of the timeseries to integer values and then the size of ouput timeseries can be reduced by specifying how much percentage to reduce
2. [ReadMetricFileFromFile.java](./ReadMetricFileFromFile.java)

    This is a modification of [DSN parser code](https://git.corp.adobe.com/AdobeResearchIndia/dsn/tree/parser/processing/src/main/java/org/processing/examples/ReadMetricFromFile.java). It is used to convert logs from protobuf to csv containing required fields. Currently it takes input a folder name that contains the protobug files for different timestamps of that day and extract some of the fields like (mesos_task_id, mesos_task_total_cpu_time, mesos_task_cpu_limit etc.).
    
    To use it, clone the [DSN repo](https://git.corp.adobe.com/AdobeResearchIndia/dsn/) and replace the [file](https://git.corp.adobe.com/AdobeResearchIndia/dsn/tree/parser/processing/src/main/java/org/processing/examples/ReadMetricFromFile.java) in parser branch with the given file. Select `File` > `Export` > `Java > Runnable JAR file` > Specify destination > `Finish`
    
    RUN `java -jar ./data_to_csv.jar input_folder_name output.csv`
3. [start_end_time.py](./start_end_time.py)

    It finds out start time and end time of each task_id and write it a csv(name given by user)

4. [ethos_time_series.ipynb](./ethos_time_series.ipynb)

    First it contains analysis of a node level utilization followed by the container level utilization running in the node for one day 27 may,2018.Conversion of long cpu and memory utilization plots for some services to smoothed plots and their smoothed plots with proper x and y labels in python matplotlib.
    
5. [arrival_pattern_analysis.ipynb](./arrival_pattern_analysis.ipynb)

    Scatter plots of some periodically arriving services , some very frequently, less frequently coming in chunks.
    Distribution of job lengths and no of jobs, hourly analysis across five days of no of task ids through box plot analysis.
    Picked up 8 distinct services which had a mix of periodic and non periodic patterns and scatter plot analysis of those.
    Heatmap of services and hour of the day with color intensity based on no of that service arriving within that hour gap.

6. [time_series_clustering.ipynb](./time_series_clustering.ipynb)

  Using tsfresh library features like mean,maximum,minimum,standard deviation etc were extracted from the time series derived from ethos production logs.All the time series which had mean value of utilisation less than 1 were filtered out. Then smoothening of filtered time series was done with hanning window of length 15. At last features from the smoothed time series were extracted which were used for clustering. Chose top k features based on principal feature analysis and the optimum no of clusters=4 and k value was decided on the basis of accuracy defined in the notebook 

## Shape-based clustering of time series using dynamic time warping

### Setup
1. Create `outputs/data` and `outputs/imgs` folder


#### Instructions for clustering
```
python3 read_ethos.py <root_path_to_csvs>
python3 cluster.py --prepare_ts --data_path test_ts_data_list.pkl -w 10 -ds 1
python3 cluster.py --compute_dtw_dist_matrix -n <num_timeseries> -w 10 -ds 1 -r 10
python3 cluster.py --cluster_ts -n <num_timeseries> -w 10 -ds 1 -r 10 -k 2,3,4,5 -it 100
python3 cluster.py --compute_kclust_error -n <num_timeseries> -w 10 -ds 1 -r 10 -k 2,3,4,5 -it 100
```
1st command dumps the timeseries, each in a csv, into `test_ts_data_list.pkl` and prints number of time_series

The 2nd command pre processes the time series.

The 3rd one computes the dyw distance matrix and dumps it in `outputs/data`

The 4th one clusters the time series and finally,

the last one computes the cluster error for the elbow plot.

#### Notes on parameters:
- w: window size for smoothing (int, e.g. int(0.1 * n); or float, e.g 0.1)
- ds: downsample rate
- n: number of samples in dataset
- r: window size for LB_Keogh (int, e.g. int(0.03 * n))
- k: comma-separated ints for values of k in k-medoids
- it: number of iterations to run k-medoids

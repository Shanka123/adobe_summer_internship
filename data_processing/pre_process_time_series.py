"""
process_ts has the following inputs:
1. time_series : The time series that needs to be processed as a list of floats.
2. MAX_Y_DIST  : Amount of variation to be made constant. default value = 4.

It converts the time_series into one with integer values. Moreover, if the time_series has consecutive values between the range of MAX_Y_DIST from the first value,
then all these are made constant and equal to their mean.

returns the processed time_series as a list
"""
def convert_ts_utilization_values_to_integer(time_series, MAX_Y_DIST = 4):
    last_constant_index, current_constant_index = 0, 0
    last_y = time_series[current_constant_index]
    max_len = len(time_series)
    print(max_len)
    x=[-1 for x in range(max_len)]
    while current_constant_index < max_len:
        summ = 0
        while(abs(time_series[current_constant_index] - last_y) <= MAX_Y_DIST):
            summ += float(time_series[current_constant_index])
            current_constant_index += 1
            if current_constant_index >= max_len:
                break
        diff = current_constant_index-last_constant_index
        avg = summ//diff
        x[last_constant_index:current_constant_index] = [avg for x in range(diff)]
        last_constant_index = current_constant_index
        if current_constant_index < max_len:
            last_y = time_series[current_constant_index]

    return x

"""
inputs:
  z: represents the input timeseries with integer values
  perc: amount of timeseries to retain(in term of number of points)
output:
  reduced time_series
"""
def shrink(z, perc):
    i = 1
    f = []
    start = 0
    while( i < len(z)):
        if z[i] != z[i-1] or i == (len(z)-1):
            f += [int(abs(max(1,z[i-1])))]*int((i-start)*perc)
            start = i
        i += 1
    return f

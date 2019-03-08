import _pickle as pickle
import numpy as np
import os, sys
import pandas as pd
ts_data = []
ids = []
i = 0
ROOT = sys.argv[1]
for c in os.listdir(ROOT):
    if c!='.ipynb_checkpoints':
        t=pd.read_csv(ROOT+'/' + c, engine='python')
        ts_data.append(t['utilization'])
        ids.append(c)
        i += 1
        
pickle.dump(ts_data, open('test_ts_data_list.pkl', 'wb'), protocol=2)
print(i)
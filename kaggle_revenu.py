import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

train = pd.read_csv('../input/train.csv')#,
                    #dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/test.csv')#,
                  # dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
print(train.shape, test.shape)
print(train.columns,'\n', test.columns)

print(train.head)
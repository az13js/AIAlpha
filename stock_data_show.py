# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv('stock_data.csv')
plt.plot(np.array(data_frame.iloc[0:400, 0]), np.array(data_frame.iloc[0:400, 1]), 'g')
plt.show()

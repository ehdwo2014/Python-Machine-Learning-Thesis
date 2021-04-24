import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('Analysis1.csv')
(print(data))
heatmap1_data = pd.pivot_table(data, values='UD', index=['RSI14'], columns='RSI2')
sns.heatmap(heatmap1_data, cmap="RdBu")
plt.show()
#Set up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import sea
#Load a seaborn dataset
crash_df=sns.load_dataset("car_crashes")
crash_df.head()

print(sns.palettes.SEABORN_PALETTES)
sns.kdeplot(crash_df['alcohol'])
plt.show()

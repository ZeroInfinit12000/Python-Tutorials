# import matplotlib and numpy as usual
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# now import pylustrator
import pylustrator

# activate pylustrator
pylustrator.start()
# Load some data
API_KEY = "c33d17d6ad546cfc58302d8906a42ece"
import fredapi

fred = fredapi.Fred(api_key=API_KEY)
data_0 = pd.DataFrame(fred.get_series("PPIACO"))
data_1 = pd.DataFrame(fred.get_series("NGDPPOT"))
data_2 = pd.DataFrame(fred.get_series("GDP"))
data=pd.concat([data_0,data_1,data_2],axis=1)
data.columns=["PPIACO","NGDPPOT","GDP"]
data=data.dropna()
data.head()
plt.plot(data)

plt.show()

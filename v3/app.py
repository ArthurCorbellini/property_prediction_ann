
from feature_engine import discretisation as dsc
import pandas as pd
from data_prep import DataPrep
import matplotlib.pyplot as plt

dp = DataPrep("rent")
discretization = dsc.EqualFrequencyDiscretiser(
    q=50, variables=["condo", "size"])

discretization.fit(dp.x_train)
train_t = discretization.transform(dp.x_train)
test_t = discretization.transform(dp.x_test)

print(train_t.head())

t1 = train_t.groupby(['condo'])['condo'].count()/len(train_t)
t2 = test_t.groupby(['condo'])['condo'].count()/len(test_t)

temp = pd.concat([t1, t2], axis=1)
temp.columns = ['train', 'test']
temp.plot.bar()
plt.xticks(rotation=0)
plt.ylabel('Number of observartion per bin')
plt.tight_layout()
plt.show()

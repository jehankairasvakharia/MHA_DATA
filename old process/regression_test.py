import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_results(x,y,d):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept

def plot_results(x, intercept, slope):
    # plt.plot(x, y, label='original data')
    plt.plot(x, intercept + slope * x, 'r', label='fitted line')
    plt.legend()
    plt.show()


data = pd.read_csv('C:/Users/Public/PYTHON_PROJECTS/MHA_data/CSVBOIS/design_1_clean.csv', header=[0,1])

print(data.head())

only3 = data[[('version_id21','version_id21')   ,   ('reward_design2', ' "This is the kind of text message I would want to receive."')]].copy()

print(only3.head())

only3['A1B1'] = only3[('version_id21','version_id21')] == 5296.0

only3['A1B2'] = only3[('version_id21','version_id21')] == 5297.0

only3['A1B3'] = only3[('version_id21','version_id21')] == 5298.0

only3['A2B1'] = only3[('version_id21','version_id21')] == 5299.0

only3['A2B2'] = only3[('version_id21','version_id21')] == 5300.0

only3['A2B3'] = only3[('version_id21','version_id21')] == 5301.0


print(only3.head(20))
reward_name = ('reward_design2', ' "This is the kind of text message I would want to receive."')

#
y = only3[reward_name]
x1 = only3['A1B1']
x2 = only3['A1B2']
x3 = only3['A1B3']
x4 = only3['A2B1']
x5 = only3['A2B2']
x6 = only3['A2B3']

#ressies
s1, i1 = get_results(x1, y, only3)
s2, i2 = get_results(x2, y, only3)
s3, i3 = get_results(x3, y, only3)
s4, i4 = get_results(x4, y, only3)
s5, i5 = get_results(x5, y, only3)
s6, i6 = get_results(x6, y, only3)



#plot
# plot_results(x1, i1, s1)
# plot_results(x2, i2, s2)
# plot_results(x3, i3, s3)
# plot_results(x4, i4, s4)
# plot_results(x5, i5, s5)
# plot_results(x6, i6, s6)

plt.plot(x1, i1 + s1 * x1, 'r', label='A1B1')
plt.plot(x2, i2 + s2 * x2, 'm', label='A1B2')
plt.plot(x3, i3 + s3 * x3, 'y', label='A1B3')
plt.plot(x4, i4 + s4 * x4, 'g', label='A2B1')
plt.plot(x5, i5 + s5 * x5, 'b', label='A2B2')
plt.plot(x6, i6 + s6 * x6, 'k', label='A2B3')

plt.ylim(1,4)
plt.legend()
plt.show()
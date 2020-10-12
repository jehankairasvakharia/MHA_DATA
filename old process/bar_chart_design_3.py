import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import os


data = pd.read_csv('C:/Users/Public/PYTHON_PROJECTS/MHA_data/CSVBOIS/design_1_clean.csv', header=[0,1])

print(data.head())

clean = data[[('version_id_3_1', 'version_id_3_1')   , ('reward_3_1', '"This is the kind of text message I would want to receive."')]].copy()

print(clean.head())

clean['C1'] = clean[('version_id_3_1', 'version_id_3_1')] == 5302.0

clean['C2'] = clean[('version_id_3_1', 'version_id_3_1')] == 5303.0

clean['C3'] = clean[('version_id_3_1', 'version_id_3_1')] == 5304.0




print(clean.head(20))
reward = ('reward_3_1', '"This is the kind of text message I would want to receive."')

#
y = clean[reward]
x1 = clean[clean['C1']][[('C1', ''), reward]]
x2 = clean[clean['C2']][[('C2', ''), reward]]
x3 = clean[clean['C3']][[('C3', ''), reward]]

count1 = x1[reward].sample(frac=0.3, random_state=2).value_counts()
count2 = x2[reward].sample(frac=0.3, random_state=2).value_counts()
count3 = x3[reward].sample(frac=0.3, random_state=2).value_counts()


counts = [count1,count2,count3]


for count in counts:
    for i in range(0,5):
        if i not in count.index:
            count._set_value(i, 0)



# X Values
x = [1,2,3,4,5]


# C1 Values
plt.subplot(1, 3, 1)
y1 = [count1[rating] for rating in range(0,5)]
plt.bar(x,y1)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y1)
Y1 = [a*slope + intercept for a in x]
plt.plot(x, Y1, 'r', label='fitted line')
plt.ylim(0,5)
plt.xlabel('rating')
plt.ylabel('counts')
plt.title('C1')


# C2 Values
plt.subplot(1, 3, 2)
y2 = [count2[rating] for rating in range(0,5)]
plt.bar(x,y2)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y2)
Y2 = [a*slope + intercept for a in x]
plt.plot(x, Y2, 'r', label='fitted line')
plt.ylim(0,5)

plt.xlabel('rating')
plt.ylabel('counts')

plt.title('C2')

# C3 Values
plt.subplot(1, 3, 3)
y3 = [count3[rating] for rating in range(0,5)]
plt.bar(x,y3)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y3)
Y3 = [a*slope + intercept for a in x]
plt.plot(x, Y3, 'r', label='fitted line')
plt.ylim(0,5)
plt.xlabel('rating')
plt.ylabel('counts')

plt.title('C3')

plt.show()

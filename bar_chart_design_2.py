import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import os


data = pd.read_csv('C:/Users/Public/PYTHON_PROJECTS/MHA_data/CSVBOIS/design_1_clean.csv', header=[0,1])

print(data.head())

clean = data[[('version_id21', 'version_id21')   , ('reward_design2', ' "This is the kind of text message I would want to receive."')]].copy()

print(clean.head())

clean['A1B1'] = clean[('version_id21', 'version_id21')] == 5296.0

clean['A1B2'] = clean[('version_id21', 'version_id21')] == 5297.0

clean['A1B3'] = clean[('version_id21', 'version_id21')] == 5298.0

clean['A2B1'] = clean[('version_id21', 'version_id21')] == 5299.0

clean['A2B2'] = clean[('version_id21', 'version_id21')] == 5300.0

clean['A2B3'] = clean[('version_id21', 'version_id21')] == 5301.0


print(clean.head(20))
reward = ('reward_design2', ' "This is the kind of text message I would want to receive."')

#
y = clean[reward]
x1 = clean[clean['A1B1']][[('A1B1', ''), reward]]
x2 = clean[clean['A1B2']][[('A1B2', ''), reward]]
x3 = clean[clean['A1B3']][[('A1B3', ''), reward]]
x4 = clean[clean['A2B1']][[('A2B1', ''), reward]]
x5 = clean[clean['A2B2']][[('A2B2', ''), reward]]
x6 = clean[clean['A2B3']][[('A2B3', ''), reward]]

count1 = x1[reward].sample(frac=0.3, random_state=2).value_counts()
count2 = x2[reward].sample(frac=0.3, random_state=2).value_counts()
count3 = x3[reward].sample(frac=0.3, random_state=2).value_counts()
count4 = x4[reward].sample(frac=0.3, random_state=2).value_counts()
count5 = x5[reward].sample(frac=0.3, random_state=2).value_counts()
count6 = x6[reward].sample(frac=0.3, random_state=2).value_counts()

counts = [count1,count2,count3,count4,count5,count6]


for count in counts:
    for i in range(0,5):
        if i not in count.index:
            count._set_value(i, 0)



# X Values
x = [1,2,3,4,5]


# A1B1 Values
plt.subplot(2, 3, 1)
y1 = [count1[rating] for rating in range(0,5)]
plt.bar(x,y1)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y1)
Y1 = [a*slope + intercept for a in x]
plt.plot(x, Y1, 'r', label='fitted line')
plt.ylim(0,5)
plt.xlabel('rating')
plt.ylabel('counts')
plt.title('A1 and B1')


# A1B2 Values
plt.subplot(2, 3, 2)
y2 = [count2[rating] for rating in range(0,5)]
plt.bar(x,y2)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y2)
Y2 = [a*slope + intercept for a in x]
plt.plot(x, Y2, 'r', label='fitted line')
plt.ylim(0,5)

plt.xlabel('rating')
plt.ylabel('counts')

plt.title('A1 and B2')

# A1B3 Values
plt.subplot(2, 3, 3)
y3 = [count3[rating] for rating in range(0,5)]
plt.bar(x,y3)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y3)
Y3 = [a*slope + intercept for a in x]
plt.plot(x, Y3, 'r', label='fitted line')
plt.ylim(0,5)
plt.xlabel('rating')
plt.ylabel('counts')

plt.title('A1 and B3')

# A2B1 Values
plt.subplot(2, 3, 4)
y4 = [count4[rating] for rating in range(0,5)]
plt.bar(x,y4)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y4)
Y4 = [a*slope + intercept for a in x]
plt.plot(x, Y4, 'r', label='fitted line')
plt.ylim(0,5)
plt.xlabel('rating')
plt.ylabel('counts')

plt.title('A2 and B1')

# A2B2 Values
plt.subplot(2, 3, 5)
y5 = [count5[rating] for rating in range(0,5)]
plt.bar(x,y5)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y5)
Y5 = [a*slope + intercept for a in x]
plt.plot(x, Y5, 'r', label='fitted line')
plt.ylim(0,5)
plt.xlabel('rating')
plt.ylabel('counts')

plt.title('A2 and B2')


# A2B3 Values
plt.subplot(2, 3, 6)
y6 = [count6[rating] for rating in range(0,5)]
plt.bar(x,y6)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y6)
Y6 = [a*slope + intercept for a in x]
plt.plot(x, Y6, 'r', label='fitted line')
plt.ylim(0,5)
plt.xlabel('rating')
plt.ylabel('counts')

plt.title('A2 and B3')

plt.show()
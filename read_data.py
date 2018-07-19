import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


df = pd.read_csv('socialcuriosity_10.csv')
df1 = df[df.columns[17:].tolist()]
df1 = df1.drop(df.index[[0, 1]])
df1 = df1.reset_index(drop = True)
df1 = df1.rename(columns= {'Q1': 'User1', 'Q2': 'User2','Q3': 'Gender', 'Q4': 'Age', 'Q5': 'Education'})
df2 = pd.DataFrame(np.zeros([len(df1), 15]), columns=['joyous_ex_self', 'deprivation_sens_self', 'stress_tol_self',
                                                    'social_cur_self', 'thrill_seek_self', 'joyous_ex_partner',
                                                    'deprivation_sens_partner', 'stress_tol_partner',
                                                    'social_cur_partner', 'thrill_seek_partner',
                                                      'joyous_ex_reflection','deprivation_sens_reflection', 'stress_tol_reflection',
                                                    'social_cur_reflection', 'thrill_seek_reflection'])



df3 = pd.concat([df1, df2], axis=1)
for i in df3:
    df3[i] = np.array(df3[i], dtype=float)
fivejump = np.arange(0, 25, 5)
for i in range(5):
    self_array = np.mean(df3[df3.columns[df3.columns.str.contains('Q6_')][fivejump + i]], axis=1)
    partner_array = np.mean(df3[df3.columns[df3.columns.str.contains('Q7_')][fivejump + i]], axis=1)
    for j in range(len(df3)):
        df3.iloc[j, len(df1.columns) + i] = self_array[j]
        df3.iloc[j, len(df1.columns) + i + 5] = partner_array[j]
        df3.iloc[j, len(df1.columns) + i + 5] = partner_array[j]


for ind in df3.index:
    temp = df3[df3.User2 == df3.loc[ind, 'User1']].iloc[:, -10:-5]
    df3.iloc[ind, -5:] = np.asarray(temp)

df3 = df3.drop(['joyous_ex_partner', 'deprivation_sens_partner', 'stress_tol_partner', 'social_cur_partner', 'thrill_seek_partner'], axis=1)
averages = pd.Series(index = df3.columns, data = np.zeros(len(df3.columns)))
stds = pd.Series(index = df3.columns, data = np.zeros(len(df3.columns)))

# for i in range(len(df3.columns)):
#     averages[i] = np.mean(df3.iloc[:, i])
#     stds[i] = np.std(df3.iloc[:, i])

averages = df3.mean()
stds = df3.std()
stds = stds[-10:]
stds = [stds[:5], stds[5:]]

array_two = [0,1]
df3.to_csv('clear_data_frame.csv')
df1.to_csv('df1_data_frame')
# print (df3)

curious_index_names = ['joyous_ex', 'deprivation_sens', 'stress_tol', 'social_cur', 'thrill_seek']
comparison = pd.DataFrame({'self': list(averages[55:60]),
                               'reflection': list(averages[60:65])},
                              index=curious_index_names)
comparison1 = df3.iloc[:,55:]

def graph_individual():

    joyous_ax = df3[df3.columns[df3.columns.str.contains('joyous_ex')][array_two]].plot.bar()
    depr_ax = df3[df3.columns[df3.columns.str.contains('deprivation_sens')][array_two]].plot.bar()
    stre_ax = df3[df3.columns[df3.columns.str.contains('stress_tol')][array_two]].plot.bar()
    socia_ax = df3[df3.columns[df3.columns.str.contains('social_cur')][array_two]].plot.bar()
    thri_ax = df3[df3.columns[df3.columns.str.contains('thrill_seek')][array_two]].plot.bar()

    joyous_ax.set_title('joyous_ex')
    depr_ax.set_title('deprivation_sens')
    stre_ax.set_title('stress_tol')
    socia_ax.set_title('social_cur')
    thri_ax.set_title('thrill_seek')

def graph_type():
    fig, ax = plt.subplots(1,1)
    comp_ax = comparison.plot(kind = 'bar', yerr = stds, ax =ax)
    comp_ax.set_title('all_types')

def graph_all():
    all_ax = np.mean(comparison).plot.bar()
    all_ax.set_title('combined')

def graph_difference():
    for c in curious_index_names:
        print(c)
        temp = df3.iloc[:, df3.columns.str.contains(c)].diff(axis=1).abs().iloc[:, -1]
        df3[c + '_diff'] = temp
    differnce_cindex = df3.iloc[:,-5:]

    fig, ax = plt.subplots(1,1)
    differnce_cindex.mean(axis=0).plot(kind= 'bar', yerr=differnce_cindex.std(axis=0), ax = ax)
    # differnce_cindex.boxplot()
    print(ttest_1samp(differnce_cindex, 0, axis=0))


graph_individual()
# graph_type()
#graph_all()
graph_difference()
plt.show()



# from IPython import embed
# embed()

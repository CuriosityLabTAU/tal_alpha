import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import seaborn as sns
import os
import scipy.stats as stats

fname = 'socialcuriosity_13.csv'
df = pd.read_csv(os.getcwd() +'/data/qualtrics_5dc/' + fname)
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
df3 = df3.drop(df3[(df3.User1 == 1001) | (df3.User1 == 1002)].index, axis=0)
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
    comparison2 = comparison/comparison.max()
    comparison2 = pd.DataFrame(comparison2.mean(axis=1), columns = ['mean'])
    # comp_ax = comparison.plot(kind = 'bar', yerr = stds, ax =ax)
    comp_ax = comparison2.plot(kind = 'bar', ax =ax)
    comp_ax.set_title('all_types')

def graph_all():
    all_ax = np.mean(comparison).plot.bar()
    all_ax.set_title('combined')

def graph_difference():
    for c in curious_index_names:
        temp = df3.iloc[:, df3.columns.str.contains(c)].diff(axis=1).abs().iloc[:, -1]
        df3[c + '_diff'] = temp
    differnce_cindex = df3.iloc[:,-5:]

    fig, ax = plt.subplots(1,1)
    differnce_cindex /= differnce_cindex.max()
    # differnce_cindex.mean(axis=0).plot(kind= 'bar', yerr=differnce_cindex.std(axis=0), ax = ax)
    differnce_cindex.mean(axis=0).plot(kind= 'bar', yerr=differnce_cindex.std(axis=0), ax = ax)
    # differnce_cindex.boxplot()
    print('==============5DC t-test==============')
    print(ttest_1samp(differnce_cindex, 0, axis=0))

    a = df3.iloc[:, df3.columns.str.contains('6_')]
    b = df3.iloc[:, df3.columns.str.contains('7_')]
    d = np.array(a) - np.array(b)
    difference_df = pd.DataFrame(data = np.abs(d), columns = df3.iloc[:, df3.columns.str.contains('6_')].columns)
    difference_array = pd.Series(np.zeros(len(difference_df.columns)), index = [difference_df.columns])
    for i in range(len(difference_array)):
        difference_array[i] = np.mean(difference_df.iloc[:, i])

    ax.set_xlabel('Dimension')
    ax.set_ylabel('Difference')
    ax.set_xticklabels(curious_index_names, rotation=0)

    fig, ax1 = plt.subplots(1, 1)
    difference_df /= difference_df.max()
    difference_df.mean(axis=0).plot(kind= 'bar', yerr=difference_df.std(axis=0), ax = ax1)
    ax1.set_title('n = ' + str(difference_df.shape[0]))
    ax1.set_xlabel('Question')
    ax1.set_ylabel('Difference')
    print('==============All questions t-test==============')
    print(ttest_1samp(difference_df, 0, axis=0))

    return differnce_cindex



graph_individual()
graph_type()
graph_all()
differnce_cindex = graph_difference()


json_df_last = pd.read_csv('json_df_last', index_col=0)

json_df_last1 = json_df_last.copy()
json_df_last1.user = json_df_last1.user + 1
json_combined = pd.concat([json_df_last, json_df_last1], axis = 0)


curious_index_names_self = []
curious_index_names_reflection = []
for n in curious_index_names:
    curious_index_names_self.append(n + '_self')
    curious_index_names_self.append(n + '_reflection')

possible = []
score = []
for i in range(7):
    possible.append('possible_' + str(i + 1))
    score.append('score_' + str(i + 1))
df_columns = ['user']
df_columns = df_columns + curious_index_names_self + curious_index_names_reflection + score + possible

combined_df = pd.DataFrame(data = np.zeros([len(df3), 25]), columns=[df_columns])
combined_df = combined_df.reset_index(drop=True)
for i in range(10):
    combined_df.iloc[:, i + 1] = list(df3.iloc[:, i - 10])

for i in range(len(combined_df)):
    combined_df.iloc[i,0] = 1003. + i

lvls = json_combined['level'].unique()
usrs = json_combined['user'].unique()
data = df3.copy()

for l in lvls:
   c = 'possible_'+str(int(l))
   data[c] = 0

for u in usrs:
    for l in lvls:
        c = 'possible_' + str(int(l))
        data.loc[data.User1 == u, c] = np.float(json_combined.loc[((json_combined.user == u) & (json_combined.level == l)), 'possible'])


data.to_csv('df_all')
data = data.drop((data.columns[data.columns.str.contains('Q6_')]) | (data.columns[data.columns.str.contains('Q7_')]), axis=1)
data.to_csv('df_filtered')

for i in range(len(json_df_last)):
    t = (combined_df['user'] == json_df_last['user'][i])
    combined_df.loc[[s for s, x in enumerate(t) if x][0]:,
    'possible_' + str(int(json_df_last['level'][i]))] = json_df_last['possible'][i]

#        json_df_last[]
#        elif json_df_last['level'] == 6:
#        elif json_df_last['level'] == 5:
#        elif json_df_last['level'] == 4:
#        elif json_df_last['level'] == 3:
#        elif json_df_last['level'] == 2:
#        elif json_df_last['level'] == 1:




#  add the possible score to the dataframe
grouped = json_combined.groupby('level')
for k, g in grouped:
    s = str(int(k)) + '_possible'
    s1 = str(int(k)) + '_score'
    differnce_cindex[s] = g['possible'].tolist()
    differnce_cindex[s1] = g['score'].tolist()

def calculate_corr_with_pvalues(df, method = 'pearsonr'):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if method == 'pearsonr':
                pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
            elif method == 'spearman':
                pvalues[r][c] = round(stats.spearmanr(df[r], df[c])[1], 4)


    rho = df.corr()
    rho = rho.round(2)
    pval = pvalues  # toto_tico's answer
    # create three masks
    r1 = rho.applymap(lambda x: '{}*'.format(x))
    r2 = rho.applymap(lambda x: '{}**'.format(x))
    r3 = rho.applymap(lambda x: '{}***'.format(x))
    # apply them where appropriate
    rho = rho.mask(pval <= 0.1, r1)
    rho = rho.mask(pval <= 0.05, r2)
    rho = rho.mask(pval <= 0.01, r3)

    return pvalues, rho

pv, corr_all = calculate_corr_with_pvalues(differnce_cindex)
print(corr_all)


plt.show()

# from IPython import embed
# embed()
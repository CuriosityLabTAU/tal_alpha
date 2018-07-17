import pandas as pd
import numpy as np

df = pd.read_csv('TalDF.csv')
df1 = df[[df.columns[8]]+ df.columns[17:].tolist()]
df1 = df1.drop(df.index[[0, 1]])
df1 = df1.reset_index(drop = True)
df1 = df1.rename(columns= {'Q1':'User1', 'Q2':'User2','Q3': 'Gender', 'Q4': 'Age', 'Q5' : 'Education'})
df1.to_csv('clear_dataframe.csv')
df2 = pd.DataFrame(np.zeros([len(df1), 10]), columns=['joyous_ex_self', 'deprivation_sens_self', 'stress_tol_self',
                                                    'social_cur_self', 'thrill_seek_self','joyous_ex_partner',
                                                    'deprivation_sens_partner', 'stress_tol_partner',
                                                    'social_cur_partner', 'thrill_seek_partner'])
df3 = df1.append(df2)
print(df3)


def curiosity_scale_average():
    joyous_ex = 0
    joyous_ex_average = 0.0
    deprivation_sens = 0
    deprivation_sens_average = 0.0
    stress_tol = 0
    stress_tol_average = 0.0
    social_cur = 0
    social_cur_average = 0.0
    thrill_seek = 0
    thrill_seek_average = 0.0
    count = 0

    print(df1.columns)
    for j in range(2, len(df1)):
        for i in range(6, (len(df1.columns)-6)//2, 5):
            # self
            df2['joyous_ex_self'][i] += int(df1.iloc[j, i])
            df2['deprivation_sens_ self'][i] += int(df1.iloc[j, i+1])
            df2['stress_tol_self'][i] += int(df1.iloc[j, i+2])
            df2['social_cur_self'][i] += int(df1.iloc[j, i+3])
            df2['thrill_seek_self'][i] += int(df1.iloc[j, i+4])
            count += 1
        for i in range((len(df1.columns) - 6) // 2 + 6, len(df1.columns), 5):
            # partner
            df2['joyous_ex_partner'][i] += int(df1.iloc[j, i])
            df2['deprivation_sens_partner'][i] += int(df1.iloc[j, i+1])
            df2['stress_tol_partner'][i] += int(df1.iloc[j, i+2])
            df2['social_cur_partner'][i] += int(df1.iloc[j, i+3])
            df2['thrill_seek_partner'][i] += int(df1.iloc[j, i+4])

    joyous_ex_average = joyous_ex / count
    deprivation_sens_average = deprivation_sens / count
    stress_tol_average = stress_tol / count
    social_cur_average = social_cur / count
    thrill_seek_average = thrill_seek / count

    print(joyous_ex_average)
    print(deprivation_sens_average)
    print(stress_tol_average)
    print(social_cur_average)
    print(thrill_seek_average)
curiosity_scale_average()

from IPython import embed
embed()
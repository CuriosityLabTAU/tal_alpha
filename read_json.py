import json
from pprint import pprint
import ast
import pandas as pd
import numpy as np
import collections
import seaborn as sns
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def jsonread(fname):
    '''
    read json file
    :param fname:
    :return: dataframe from the data of the level statistics in this game
    '''
    json_df = pd.DataFrame(np.zeros([7, 7]), columns= ['user', 'level', 'possible', 'score', 'difference', 'mean','time'])
    with open(fname) as f:
        data = json.load(f)
    for i in data:
        data[i] = json.loads(data[i])

    data = collections.OrderedDict(sorted(data.items()))

    dvalues = list(data.values()) # todo: fix this
    filtered_scores = list(filter(lambda x: 'user_score' in x['comment'], dvalues))
    filtered_times  = list(map(lambda x: x['time'], filtered_scores))
    filtered_scores = list(map(lambda x: x['comment'], filtered_scores))
    time_diff_list = [0]*7


    for i in range(len(filtered_times)):
        filtered_times[i] = int(filtered_times[i].replace('_', '')[8:-6])
        time_diff = 0
        if (i!=0):
            time_diff = filtered_times[i]%100 - filtered_times[i-1]%100
            time_diff += 60*(filtered_times[i]%10000/100 - filtered_times[i-1]%10000/100)
            time_diff += 3600*(filtered_times[i]%1000000/10000 - filtered_times[i-1]%1000000/10000)
            time_diff_list[i-1] = time_diff
            time_diff_list[6] += time_diff




    for i in range(len(filtered_scores)):
        filtered_scores[i] = json.loads(filtered_scores[i].replace("'", '"'))
    for i in range(len(filtered_scores)):
        json_df.iloc[i, 1] = i + 1
        json_df.iloc[i, 2] = filtered_scores[i]['possible_score']
        json_df.iloc[i, 3] = filtered_scores[i]['user_score']
        json_df.iloc[i, 4] = json_df.iloc[i, 2] - json_df.iloc[i, 3]
        json_df.iloc[i, 5] = json_df.iloc[i, 2:4].mean()
        json_df.iloc[i, 6] = time_diff_list[i]
        # if json_df.iloc[i, 2] - json_df.iloc[i, 3] >= 0:
        #     json_df.iloc[i, 4] = json_df.iloc[i, 2] - json_df.iloc[i, 3]
        # else:
        #     json_df.iloc[i, 4] = 0

    json_df['user'] = data[list(data.keys())[8]]['comment']

    return json_df

def main():
    for file in os.listdir(os.getcwd() + '/data/json_files'):
        if 'df_json' in locals():
            df_json = pd.concat([df_json, jsonread('data/json_files/'+file)], axis=0)
        else:
            df_json = jsonread('data/json_files/'+file)
    json_df_last = df_json.sort_values(by=['user'])
    json_df_last = json_df_last.reset_index(drop = True)

    json_df_last[['possible', 'score']] /= json_df_last[['possible', 'score']].max().max() # normaliztion

    json_df_last = json_df_last.drop(json_df_last.index[range(7)])
    json_df_last = json_df_last.reset_index(drop = True)
    json_df_last.to_csv('data/json_df_last')

    fig, ax = plt.subplots(1,1)
    sns.barplot(x='level', y='possible', hue='user', data=json_df_last)
    fig, ax = plt.subplots(1,1)
    sns.barplot(x='level', y='possible', data=json_df_last)
    fig, ax = plt.subplots(1,1)
    sns.barplot(x='level', y='score', data=json_df_last)

    print(json_df_last)
    print('possible t-test:',ttest_ind(json_df_last[json_df_last['level'] == 1.]['possible'], json_df_last[json_df_last['level'] == 2.]['possible'], axis=0))
    print('score t-test:',ttest_ind(json_df_last[json_df_last['level'] == 1.]['score'], json_df_last[json_df_last['level'] == 2.]['score'], axis=0))

    plt.show()
    # print json_df_last

if __name__ == '__main__':
    main()
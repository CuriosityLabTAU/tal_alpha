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



def jsonread(data):
    json_df = pd.DataFrame(np.zeros([7, 5]), columns= ['user', 'level', 'possible', 'score', 'difference'])
    with open(data) as f:
        data = json.load(f)
    for i in data:
        data[i] = json.loads(data[i])

    data = collections.OrderedDict(sorted(data.items()))

    dvalues = data.values()
    filtered_scores = filter(lambda x: 'user_score' in x['comment'], dvalues)
    filtered_times  = map(lambda x: x['time'], filtered_scores)
    filtered_scores = map(lambda x: x['comment'], filtered_scores)

    for i in xrange(len(filtered_scores)):
        filtered_scores[i] = json.loads(filtered_scores[i].replace("'", '"'))
    for i in range(len(filtered_scores)):
        json_df.iloc[i, 1] = i + 1
        json_df.iloc[i, 2] = filtered_scores[i]['possible_score']
        json_df.iloc[i, 3] = filtered_scores[i]['user_score']
        json_df.iloc[i, 4] = json_df.iloc[i, 2] - json_df.iloc[i, 3]
        # if json_df.iloc[i, 2] - json_df.iloc[i, 3] >= 0:
        #     json_df.iloc[i, 4] = json_df.iloc[i, 2] - json_df.iloc[i, 3]
        # else:
        #     json_df.iloc[i, 4] = 0

    json_df['user'] = data[data.keys()[8]]['comment']

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
    json_df_last.to_csv('json_df_last')

    fig, ax = plt.subplots(1,1)
    sns.barplot(x='level', y='possible', hue='user', data=json_df_last)
    fig, ax = plt.subplots(1,1)
    sns.barplot(x='level', y='possible', data=json_df_last)
    fig, ax = plt.subplots(1,1)
    sns.barplot(x='level', y='score', data=json_df_last)

    print json_df_last
    print('possible t-test:',ttest_ind(json_df_last[json_df_last['level'] == 1.]['possible'], json_df_last[json_df_last['level'] == 2.]['possible'], axis=0))
    print('score t-test:',ttest_ind(json_df_last[json_df_last['level'] == 1.]['score'], json_df_last[json_df_last['level'] == 2.]['score'], axis=0))

    plt.show()
    # print json_df_last

if __name__ == '__main__':
    main()
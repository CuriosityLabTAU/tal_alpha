import pandas as pd
import string

df = pd.read_csv('TalDF2.csv')
df1 = df[[df.columns[8]]+ df.columns[17:].tolist()]
df1 = df1.rename(columns= {'Q1':'User1', 'Q2':'User2','Q3': 'Gender', 'Q4': 'Age', 'Q5' : 'Education'})
df1.to_csv('clear_dataframe.csv')
print (df1)
print(df1.columns)
#Gender = pd.Series(df1['Gender'])
#for i in range(len(Gender)):
#    try:
#        if int(Gender[i].strip()) == 1:
#            Gender[i] = 'Female'
#        elif int(Gender[i].strip()) == 2:
#            Gender[i] = 'male'
#        elif int(Gender[i].strip()) ==3:
#            Gender[i] = 'other'
#    except:
#        pass
#
#df1['Gender'] = Gender
#print df1['Gender']
#Q1Answers = pd.Series(df1['Q4_1'])
#print Q1Answers
#GenderToQ1 = pd.DataFrame(data = 0, index = ('Female', 'Male', 'Other'), columns= range(1,6))
#for i in range(11):
#    try:
#        if df1['Gender'][i] == 'Male':
#            GenderToQ1['Male'][int(Q1Answers[i].strip())] += 1
#        elif df1['Gender'][i] == 'Female':
#            GenderToQ1['Female'][int(Q1Answers[i].strip())] += 1
#        elif df1['Gender'][i] == 'Other':
#            GenderToQ1['Other'][int(Q1Answers[i].strip())] += 1
#    except:
#        pass
#print GenderToQ1

UserDiffernce = pd.DataFrame(data = 0, index= range(1,6), columns = ('User1', 'User2'))
print UserDiffernce
for i in range(1, len(UserDiffernce)):
    UserDiffernce['User2'][i]=df1.loc[df1.User1 == i, 'Q7_1']
    UserDiffernce['User1'][i]=df1.loc[df1.User1 == df1.loc[df1.User1 == i, 'User2'], 'Q6_1']


import json
from pprint import pprint

with open('2018_07_19_14_09_20_905198.log') as f:
    data = json.load(f)


pprint(data)

#from IPython import embed
#embed()
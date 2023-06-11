import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "store_data.csv")
#print(my_path)

store_data = pd.read_csv(path, header = None)


#print(store_data.head())
records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])
#print("Records:",records)
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

#print(association_results)
#print(association_rules)
#for i in range(1,len(association_results)):
    #print(association_results[i])
#item_bought = input("Enter the item already bought:")
def get_recommend(item_bought):
    for item in association_results:
        #print("**********")
        pair = item[0]
        items = [x for x in pair]
        #print("Rule: " + items[0] + " -> " + items[1])

        #second index of the inner list
        #print("Support: " + str(item[1]))

        #third index of the list located at 0th
        #of the third index of the inner list

        #print("Confidence: " + str(item[2][0][2]))
        #print("Lift: " + str(item[2][0][3]))
        #print("=====================================")
        #print("Item is:",items[0])
        #print("Input is:",item_bought)
        if (items[0] == item_bought):
            #print("Recommended Item: ",items[1])
            #print("Support: " + str(item[1]))
            #print("Confidence: " + str(item[2][0][2]))
            #print("Lift: " + str(item[2][0][3]))
            return items[1], str(item[2][0][2])
            break
        #print("=====================================")


# recommended_output = get_recommend("olive oil")
# print("Recommend:",recommended_output[0])
# print("Percentage:",round(float(recommended_output[1]),2)*100)

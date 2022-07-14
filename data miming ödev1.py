# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 01:33:01 2022

@author: erenk
"""


import pyfpgrowth
import pandas as pd
import numpy as np

#here I will use the association rules to find which kind of product is sold in which city
#for example if trucks are sold much in a city it means there is a construction boom etc so
#to make better offers we can use that information
#DEALSİZE-PRODUCTLİNE
#productline city
#CITY-DEALSIZE


data = pd.read_csv("sales_data_sample1.csv", sep=",", encoding='Latin-1')
data = pd.DataFrame(data)


df_productline = pd.DataFrame(data=data, index = range(2822), columns = ['PRODUCTLINE'])
df_city = pd.DataFrame(data=data, index = range(2822), columns = ['CITY'])
df_dealsize = pd.DataFrame(data=data, index = range(2822), columns = ['DEALSIZE'])
df_ordernumber = pd.DataFrame(data=data, index = range(2822), columns = ['ORDERNUMBER'])


df_productcity = pd.concat([df_productline,df_city], axis=1)
df_productcity['PRODUCTLINE'] = df_productcity[['PRODUCTLINE', 'CITY']].apply(lambda x: ','.join(x), axis=1)
new_df=df_productcity['PRODUCTLINE'] = df_productcity.PRODUCTLINE.apply(lambda x: x.split(' '))


df_dealproduct = pd.concat([df_productline,df_dealsize], axis=1)
df_dealproduct['PRODUCTLINE'] = df_dealproduct[['PRODUCTLINE', 'DEALSIZE']].apply(lambda x: ','.join(x), axis=1)
new_df2=df_dealproduct['PRODUCTLINE'] = df_dealproduct.PRODUCTLINE.apply(lambda x: x.split(' '))

df_citydeal = pd.concat([df_city,df_dealsize], axis=1)
df_citydeal['CITY'] = df_citydeal[['CITY', 'DEALSIZE']].apply(lambda x: ','.join(x), axis=1)
new_df3 =df_citydeal['CITY'] = df_citydeal.CITY.apply(lambda x: x.split(' '))

"""
df_productorder = pd.concat([df_productline,df_ordernumber], axis=1)
df_productorder['ORDERNUMBER'] = df_productorder[[ 'ORDERNUMBER','PRODUCTLINE']].apply(lambda x: ','.join(x), axis=1)
new_df4=df_productorder['ORDERNUMBER'] = df_productorder.ORDERNUMBER.apply(lambda x: x.split(' '))
"""

patterns = pyfpgrowth.find_frequent_patterns(transactions=new_df,support_threshold=0.5)
rules = pyfpgrowth.generate_association_rules(patterns, .5)

patterns2 = pyfpgrowth.find_frequent_patterns(new_df2,2)
#rules = pyfpgrowth.generate_association_rules(patterns2, .5)

patterns3 = pyfpgrowth.find_frequent_patterns(new_df3,2)
#Rules=pyfpgrowth.generate_association_rules(patterns=patterns3,confidence_threshold=0.5)
#here we see the patterns, which city ordered which etc. 


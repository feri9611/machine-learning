# %%
from unicodedata import category
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import pearsonr 
from scipy import stats
from sklearn import preprocessing


smartphones = pd.read_csv('C://smartphones.csv')
# pearsons_coeffficent , _ = pearsonr(smartphones.inch , smartphones.weight)
# pearsons_coeffficent


# %% [markdown]
# num_var = smartphones.drop(['name','os','capacity','Ram','company', axis=1])
# corr = num_var.corr()
# 
# sb.heatmap(corr , xticklabels=corr.columns,yticklabels=corr.columns,vmin=-1,vmax=1)
# plt.show()
# 
# 

# %%
# num_var = smartphones.drop(['name','os','capacity','Ram','company'], axis=1)
# corr = num_var.corr()
# corr
# heatmap = sb.heatmap(corr , xticklabels=corr.columns,yticklabels=corr.columns,vmin=-1,vmax=1)
# plt.show()
# cat_var = smartphones.drop(['name','os','weight','inch','company'],axis=1)
# cat_var
from scipy.stats import spearmanr
#spearmanr_coefficent, _ = spearmanr(cat_var.capacity,cat_var.Ram)
#spearmanr_coefficent
# pearsonr_coefficent , _ = pearsonr(cat_var.capacity,cat_var.Ram)
# pearsonr_coefficent


# %% [markdown]
# #WHAT IS P-VALUE?

# %%
from scipy.stats import chi2_contingency
table = pd.crosstab(smartphones.capacity, smartphones.Ram)
chi2, p_value , dof , expected = chi2_contingency(table.values)
#chi2
#p_value
# dof
# expected


# %% [markdown]
# IF CHI2(in specefic value of dof for example in this example the value of dof is 9) 
# IS UPPER THAN P-VALUE THEN WE CANT REJECT NULL HYPOTHESIS 
# CHI2 > P-VALUE ----->>> WE ARE ALMOST IN NULL HYPOTHESIS REGION AND WE CANT REJECT NULL!!!!!!!!!

# %%
#  def ECDF(data):
#       n= len(data)
#       x= np.sort(data)
#       y= np.arange(1, n+1)/n
#       return x,y
# x,y = ECDF(smartphones.inch)
# plt.figure(figsize=(11,8))
# plt.scatter(x , y , s=80 )
# plt.margins(0.05)
# plt.xlabel('inch',fontsize=15)
# plt.ylabel('ECDF',fontsize=15)
smartphones

# %% [markdown]
# BERNOULI SIMULATION

# %% [markdown]
# PSEUDO RANDOM NUMBER GENERATOR

# %%
np.random.seed(666666666)
rand_num = np.random.random(3)
rand_num
#win = rand_num > 0.5
# rand_num[win]
# win(VALUES THAT ARE HIGHER THAN 0.5)

# %% [markdown]
# PROBIBALITY FOR SIMLUATION OF RANDOM NUMBERS

# %%
num_trial = 1000
rand_num = np.random.random(size=num_trial)
win = rand_num > 0.5
num_win = np.sum(win)
num_win/num_trial   

# %% [markdown]
# GAME!!!!

# %%
step = 0 
dice = np.random.randint(1, 7)
 
if dice < 3 : 
     step = max(0, step - 1)
elif dice <=5 :
    step = step + 1

else :
    num= np.random.randint(1, 7)
    step = step + num

print('dice is {} and you are in step {}'. format(dice , step))
#   ^-^

# %% [markdown]
# RANDOM WALK

# %%
step = 0 
rand_walk = np.empty(0)


for i in range(100) :
    
    dice = np.random.randint(1, 7)

    if dice < 3 : 
          step = max(0, step - 1)
    elif dice <=5 :
        step = step + 1

    else :
         num= np.random.randint(1, 7)
         step = step + num
    rand_walk = np.append(rand_walk , step)
    #print('dice is {} and you are in step {}'. format(dice , step))


# %%
# d= np.empty(0)
# for n_randwalk in range(1000):
#     step = 0 
#     rand_walk = np.empty(0)


#     for i in range(100) :
        
#         dice = np.random.randint(1, 7)

#         if dice < 3 : 
#             step = max(0, step - 1)
#         elif dice <=5 :
#             step = step + 1

#         else :
#             num= np.random.randint(1, 7)
#             step = step + num
#     d = np.append(d , step)
        
#         #print('dice is {} and you are in step {}'. format(dice , step))



# %%
# d.size
# # plt.Figure(size = (10,5))
# # sb.distplot(d)
# # plt.show()

# %%
# import seaborn as sb 
# plt.figure(figsize = (10,5))
# sb.distplot(d)
# plt.show()
# np.mean(d>60)

# %% [markdown]
# NORMAN DISTRIBUTION

# %%
# samples = np.random.normal(2, 1, size = 1000000)
# sb.distplot(samples)
# plt.show()

# %%
#plt.step(np.arange(100),rand_walk)
#plt.show()

# %% [markdown]
# **PREPROCESSING**

# %%
country = pd.read_csv('D://bak.csv' , encoding='ansi' , header= 0)
country = country.rename(columns={'Country Name':'Name', "Country Code":'code', 'Population growth':'pop_growth','Total population':'pop'})
#we make an deactionary for country's features to go along easier 
#country.shape
country.drop('code' ,axis=1 , inplace = True)
country.rename(index=country.Name , inplace=True)
country.drop('Name',axis = 1 , inplace = True)
# country.drop('0',axis=1,inplace=True)
# country_new = country(['pop','Area'])
# print(country_new)
# country.filter(['pop','Area','Population growth '], axis=1))
# new_country=country.dropna(how'all')
# new_country
new = country
country 

# %%
new = country.filter(['pop','Area','Population growth '], axis=1)
new.dropna(inplace= True,how='all')
country =  new
# for i in range(len(country)):
# print(float(country.iat[2, 0]))

for i in range(len(country)):
    for j in range(0,2):
        country.iat[i,j] = float(country.iat[i,j])
type(country.iat[0,2])

# %%
country
# new.info()
# country
# country.describe()


# %%
    # str('pop')
max_pop = country['pop'].max()
country['pop'][country['pop']==max_pop]

# %%
# country.describe()
# country.drop('World', axis=0, inplace = True)
# country
# country = country.replace(['NaN'], '3.495070')
country.isnull().sum()
# print(country.iat[13,2])

# %%
country

# %% [markdown]
# FILLING NAN(IS NOT GOOD WAY TO IGNORE NUN VALUES)

# %%
# country.fillna(0)
# country.fillna({'Population growth ':0 , 'pop':1000000 ,'Area':50000})
# country.fillna(method='ffill')
from sklearn.preprocessing import imputer
# from sklearn.impute import SimpleImputer
# import numpy as np
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean'9)
imp=imputer(missing_value ='NaN',strategy='mean',axis = 0)
# imp.fit(country)
# new_daraset= imp.transfor(country)
# new_daraset



# %% [markdown]
# BELEKHARE JAVAB DAD VALI BA RAVESHE JADID

# %%
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(country)
SimpleImputer()
X = [country]
print(imp_mean.transform(country))

# %% [markdown]
# DUPPLICATES

# %%
my_sourse1 = np.array([['babak','1','13','17','17'],['raha','2','14','14','15'],['sara','3','17','12','20'],['reza','4','12','18','19']])
my_sourse11=pd.DataFrame(my_sourse1,index=[0,1,2,3],columns=[0,1,2,3,4])
my_sourse2 = np.array([['babak','1','13','17'],['baran','2','15','20'],['sara','3','14','19'],['arash','4','20','19'],['mahan','5','15','18'],['reza','6','14','12']])
my_sourse22=pd.DataFrame(my_sourse2,index=[0,1,2,3,4,5],columns=[0,1,2,3])



# %%
my_concat = pd.concat([my_sourse11,my_sourse22], axis = 0, ignore_index=True)
my_concat.drop([4],axis=1,inplace=True)
my_concat
my_concat.drop_duplicates(inplace=True)
my_concat
my_concat.reset_index(inplace=True,drop=True)
my_concat


# %%
smartphones 
# smartphones.describe()
# smartphones.os.value_counts()

# %% [markdown]
# miangin ro bhmun bre har os nshun mide

# %%
cat_os = smartphones.groupby(smartphones['os'])
cat_os.mean()

# %%
# pd.crosstab(smartphones.os,smartphones.capacity)
smartphones

# %%

# pd.pivot_table(smartphones,index="name", columns='company',values='Ram')


smartphones

# %%
# smartphones.rename(index=smartphones.name, inplace=True)
# smartphones.drop([' name ',' company '], axis=1, inplace=True)
# smartphones.Name
# pd.get_dummies()
smartphones
smartphones.iat[0,3]
list(smartphones.columns.values)


# %%
smartphones_data=pd.get_dummies(smartphones)
smartphones_data

# %%
from sklearn.preprocessing import scale, normalize , minmax_scale
scale_data = scale(smartphones_data)
scale_data

# %%
df_data =pd.DataFrame(scale_data,
                      index=smartphones_data.index, 
                      columns=smartphones_data.columns)
df_data

# %%
scale_data1 =  normalize(smartphones_data , norm='l1',axis=0)
df_data =pd.DataFrame(scale_data1,
                      index=smartphones_data.index, 
                      columns=smartphones_data.columns)
df_data

# %%
scale_data2 =  minmax_scale(smartphones_data ,feature_range=(0,1))
df_data =pd.DataFrame(scale_data2,
                      index=smartphones_data.index, 
                      columns=smartphones_data.columns)
df_data



# %% [markdown]
# 
# 
# 
# 
# ######## SUPERVISED LEARNING ########

# %%
from sklearn import  datasets
iris = datasets.load_iris()

# %%
iris.data.shape

# %%
iris.feature_names

# %%
iris.target_names

# %%
iris.DESCR

# %%
# iris.data
iris_df = pd.DataFrame(iris.data , columns = iris.feature_names)
iris_df

# %%
iris_df['target'] = iris.target
iris_df



# %%
#Visual EDA
iris_df['target'] = iris.target

pd.plotting.scatter_matrix(iris_df , c=iris.target , s = 150 , figsize=  [11,11]   )
plt.show()

# %% [markdown]
# KNN : K-Nearest Neighbors

# %%
x = iris.data[ : , [2,3]]
y = iris.target
plt.scatter(x[:, 0], x[:,1],c=y)
plt.show()

# %% [markdown]
# FIT

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 6 ,metric='minkowski',p=2)

x=iris.data
y=iris.target

knn.fit(x,y)    


# %% [markdown]
# Distance

# %% [markdown]
# EUCLIDEN DISTANCE

# %%
knn = KNeighborsClassifier(n_neighbors= 6 ,metric='minkowski',p=2)
# knn.fit()
x_new = np.array([[5,3,1,0.2]])
y_new = knn.predict(x_new)
y_new



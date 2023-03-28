### EX -02 OUTLIER

# AIM
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
 
# EXPLANATION
   
 An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.  

# ALGORITHM

### STEP 1
Read the given Data.

### STEP 2
Get the information about the data.

### STEP 3
Detect the Outliers using IQR method and Z score.

### STEP 4
Remove the outliers.

### STEP 5
Plot the datas using Box Plot.

# PROGRAM

```
Developed by : Mohamed islam A
Registration Number : 212220220025.
```
```
import pandas as ps
import numpy as np
import seaborn as sns

df=ps.read_csv("bhp.csv")
df

df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape

sns.boxplot(x="price_per_sqft",data=df)

q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)

IQR=q3-q1 #INTERQUARTILE RANGE
ul =q3+0.5*IQR
ll =q1-1.5*IQR

df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1

df1.shape

sns.boxplot(x='price_per_sqft',data=df1)

from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2

print(df2.shape)

sns.boxplot(x='price_per_sqft',data=df2)

df3=ps.read_csv('height_weight.csv')
df3

df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape

sns.boxplot(x='weight',data=df3)

q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)

IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR

df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4

df4.shape

sns.boxplot(x='height',data=df4)
```

# OUTPUT

### DATASET FOR BHP_CSV
![image](https://user-images.githubusercontent.com/93427248/203038580-a202f91e-c3ea-489b-9474-2889c437a626.png)
### DATASET HEAD(BHP)
![image](https://user-images.githubusercontent.com/93427248/203038614-980afe54-32cf-4403-94ee-58caf66f5542.png)
### DATASET DESCRIBE(BHP)
![image](https://user-images.githubusercontent.com/93427248/203038646-c02dd55f-2ff5-4c73-b5b0-40bb91e41174.png)
### DATASET INFO(BHP)
![image](https://user-images.githubusercontent.com/93427248/203038688-deacef2d-9c38-4b12-9213-d216d426082b.png)
### DATASET NULL VALUES(BHP)
![image](https://user-images.githubusercontent.com/93427248/203038759-94fe0759-6ec8-4790-b527-7468b926411a.png)
### DATASET SHAPE WITH OUTLIERS(BHP)
![image](https://user-images.githubusercontent.com/93427248/203038791-ffccc14d-ba35-4b5b-b4eb-24f0a091a807.png)
### DATASET BOXPLOT WITH OUTLIERS(BHP)
![image](https://user-images.githubusercontent.com/93427248/203038842-db982036-2a93-457f-87af-d3fd2ef4b3d5.png)
### DATASET WITHOUT OUTLIERS(BHP)
![image](https://user-images.githubusercontent.com/93427248/203038949-201d00c6-c4b1-456e-aa42-32fdc43bd3f1.png)

![image](https://user-images.githubusercontent.com/93427248/203038986-cb662a49-b008-4ce7-a2a7-4ffc9e52083b.png)
### DATASET SHAPE WITHOUT OUTLIERS(BHP)
![image](https://user-images.githubusercontent.com/93427248/203039036-d781aefe-a086-4c0b-b4c9-b882bb949632.png)
### DATASET BOXPLOT WITHOUT OUTLIERS(BHP)
![image](https://user-images.githubusercontent.com/93427248/203039073-4e22e7b3-ceb1-48c5-b4c8-ca5c9569ffb7.png)
### DATASET AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![image](https://user-images.githubusercontent.com/93427248/203039125-191c2995-d5a4-4d3c-9882-db5010db33d2.png)
### DATASET SHAPE AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![image](https://user-images.githubusercontent.com/93427248/203039165-c4df23ad-1761-4a30-a9d7-3bde8cff5537.png)
### DATASET BOXPLOT AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![image](https://user-images.githubusercontent.com/93427248/203039210-5bb73f2d-64af-4843-8b1c-4159576fd74b.png)
### DATASET FOR WEIGHT_HEIGHT_CSV
![image](https://user-images.githubusercontent.com/93427248/203039252-4451fb18-6ad4-49ac-8ea3-c9bd1c66a447.png)
### DATASET HEAD(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039296-a8b078e5-0804-4c75-a0c2-3bbb5542f5cd.png)
### DATASET INFO(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039328-a3a3aa36-836f-4686-b969-d13c1feb2137.png)
### DATASET DESCRIBE(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039366-c3a557db-0aa7-4c2f-aaab-1bd712819829.png)
### DATASET NULL VALUES(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039400-573c9715-2b3d-463e-94bd-8b9a81f2b004.png)
### DATASET BOXPLOT WITH OUTLIERS(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039473-502e05d0-d638-4b99-b623-da47db0f598a.png)
### DATASET AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039539-30e9488d-4421-41cf-a0ab-163e159e6550.png)

![image](https://user-images.githubusercontent.com/93427248/203039589-06a38027-a11a-4146-b589-a8fec24cad0d.png)
### DATASET SHAPE(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039630-50d21833-7c00-4bbc-816f-d3965b18b243.png)
### DATASET BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)
![image](https://user-images.githubusercontent.com/93427248/203039661-b8022ff5-b76b-4c2f-9cb0-f24b11bd69c5.png)

# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.

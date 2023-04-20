import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#PART 1
# Read the CSV file into a DataFrame
df = pd.read_csv('hw1data.csv')

# count the number of missing values in each column
num_missing = df.isnull().sum()
#print(num_missing)

#remove the following columns because they have thousands of missing values 
rm_list = ['TOTALAREA_MODE','HOUSETYPE_MODE','EXT_SOURCE_1','EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_YEAR']
#impute the following column becasue ony has 53 missing values:
impute_list=['EXT_SOURCE_2']

#imputing for exit source 2
df['EXT_SOURCE_2'] = df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].mean())

df = df.drop(rm_list, axis=1)
headers = list(df.columns)



#PART 2
num_cols = df.select_dtypes(include=['int64', 'float64'])

#plot histograms of everything before normalizing the data
length= len(num_cols.columns)

fig, axs = plt.subplots()
axs = [fig.add_subplot(length, 1, i) for i in range(1, length+1)]

for i, col in enumerate(num_cols.columns):
    axs[i].hist(num_cols[col], bins=30)
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Frequency')

plt.show()

#create a transformed dataframe
df_transformed = df.copy()
#only include the numerical data in new dataframe
df_transformed = df_transformed.select_dtypes(include=['int64', 'float64'])

#deal with positively skewed data
pos_skew=[]
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].skew() > 1:
        pos_skew.append(col)

#remove CNT_CHILDREN because it is binary 0/1 so it doesn't make sense to nomalize it 
pos_skew.remove("CNT_CHILDREN")
pos_skew.remove("TARGET")

#log transformation on the positively skewed columns
df_transformed[pos_skew] = np.log(df_transformed[pos_skew]+ 1e-10)

#deal with negatively skewed data
neg_vals=['DAYS_EMPLOYED', 'DAYS_BIRTH']
fractions=['EXT_SOURCE_2']
neg_skew=['DAYS_EMPLOYED', 'DAYS_BIRTH','EXT_SOURCE_2']

#normalize the negative values using their absolute value and log normalization 
df_transformed[neg_vals] = np.log(df_transformed[neg_vals].abs() + 1e-10)

#normalize the exit source 2 column by using min/max normalization
df_transformed['EXT_SOURCE_2'] = (df_transformed['EXT_SOURCE_2'] - df_transformed['EXT_SOURCE_2'].min()) / (df_transformed['EXT_SOURCE_2'].max() - df_transformed['EXT_SOURCE_2'].min())

#plot a histogram for each of the columns after everything has been normalized
length= len(df_transformed.columns)

fig, axs = plt.subplots()
axs = [fig.add_subplot(length, 1, i) for i in range(1, length+1)]

for i, col in enumerate(df_transformed.columns):
    axs[i].hist(df_transformed[col], bins=30)
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Frequency')

plt.show()



#PART 3
#create box plots for each column
for j in num_cols:
    df_transformed.boxplot(column=[j])
    plt.title(f'Box Plot of {j}')
    plt.ylabel('Values')
    plt.show()

# #get rid of outliers for each: state what you are doing and go for that

means = df_transformed.mean()
stds = df_transformed.std()

# drop values more than 4 standard deviations away from the mean
df_transformed = df_transformed[(np.abs(df_transformed - means) <= 2 * stds).all(axis=1)]

# print the cleaned DataFrame
print(df_transformed)

for j in num_cols:
    df_transformed.boxplot(column=[j])
    plt.title(f'Box Plot of {j}')
    plt.ylabel('Values')
    plt.show()

#for target, generate seperate boxplots for Target=0 and target=1: for each feature plot two boxplots 
#look at the subplot 

# Filter data to create two separate data frames for each TARGET value
df_target_0 = df_transformed.loc[df_transformed['TARGET'] == 0]
df_target_1 = df_transformed.loc[df_transformed['TARGET'] == 1]

#print(df_transformed.head())

# Create a figure with two subplots
fig, axes = plt.subplots(1,7, figsize=(10, 6), sharey=True)

# Plot boxplot for TARGET=0 on the first subplot
axes[0].boxplot([df_target_0['CNT_CHILDREN'],df_target_1['CNT_CHILDREN']], labels=['TARGET=0','TARGET=1'])
axes[1].boxplot([df_target_0['CNT_FAM_MEMBERS'],df_target_1['CNT_FAM_MEMBERS']], labels=['TARGET=0','TARGET=1'])
axes[2].boxplot([df_target_0['AMT_INCOME_TOTAL'],df_target_1['AMT_INCOME_TOTAL']], labels=['TARGET=0','TARGET=1'])
axes[3].boxplot([df_target_0['AMT_CREDIT'],df_target_1['AMT_CREDIT']], labels=['TARGET=0','TARGET=1'])
axes[4].boxplot([df_target_0['DAYS_EMPLOYED'],df_target_1['DAYS_EMPLOYED']], labels=['TARGET=0','TARGET=1'])
axes[5].boxplot([df_target_0['DAYS_BIRTH'],df_target_1['DAYS_BIRTH']], labels=['TARGET=0','TARGET=1'])
axes[6].boxplot([df_target_0['EXT_SOURCE_2'],df_target_1['EXT_SOURCE_2']], labels=['TARGET=0','TARGET=1'])

# Add titles and axis labels to the plots
axes[0].set_title('CNT_CHILDREN')
axes[1].set_title('CNT_FAM_MEMBERS')
axes[2].set_title('AMT_INCOME_TOTAL')
axes[3].set_title('AMT_CREDIT')
axes[4].set_title('DAYS_EMPLOYED')
axes[5].set_title('DAYS_BIRTH')
axes[6].set_title('EXT_SOURCE_2')

axes[0].set_ylabel('VALUE')
axes[1].set_ylabel('VALUE')
axes[2].set_ylabel('VALUE')
axes[3].set_ylabel('VALUE')
axes[4].set_ylabel('VALUE')
axes[5].set_ylabel('VALUE')
axes[6].set_ylabel('VALUE')

plt.show()



#PART 4
# Count the number of applicants with different housing types
housing_counts = df['NAME_HOUSING_TYPE'].value_counts()

# Plot the counts as a bar plot
housing_counts.plot(kind='bar', rot=0)
plt.title('Number of Applicants by Housing Type')
plt.xlabel('Housing Type')
plt.ylabel('Count')
plt.show()

# count the number of applicants by family status and housing type
housing_family_counts = df.groupby(['NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS']).size()

# create a pivot table with housing types as rows, family statuses as columns, and counts as values
housing_family_counts = housing_family_counts.reset_index(name='count')
housing_family_counts = housing_family_counts.pivot(index='NAME_HOUSING_TYPE', columns='NAME_FAMILY_STATUS', values='count')

# create a stacked bar plot
housing_family_counts.plot(kind='bar', stacked=True)

# set the axis labels and title
plt.xlabel('Housing Type')
plt.ylabel('Count')
plt.title('Applicants by Housing Type and Family Status')

plt.xticks(fontsize=4)

# show the plot
plt.show()

# create a dictionary of dataframes, one for each education level
edu_levels = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education']
df_dict = {edu: df[df['NAME_EDUCATION_TYPE']==edu][['AMT_INCOME_TOTAL']] for edu in edu_levels}

# create a figure with subplots for each education level
fig, axes = plt.subplots(nrows=1, ncols=len(edu_levels), figsize=(15,5), sharey=True)

# plot the boxplots for each education level in the corresponding subplot
for i, edu in enumerate(edu_levels):
    df_dict[edu].plot(kind='box', ax=axes[i], title=edu)
    axes[i].set_xlabel('')
    
plt.tight_layout()
plt.show()



#PART 5
# Creating a new column AGE from DAYS_BIRTH by dividing the entries by 365
df['AGE'] = abs(df['DAYS_BIRTH']) / 365

# Creating a new column AGE_GROUP based on the AGE column
def get_age_group(age):
    if age >= 19 and age < 25:
        return 'Very_Young'
    elif age >= 25 and age < 35:
        return 'Young'
    elif age >= 35 and age < 60:
        return 'Middle_Age'
    else:
        return 'Senior_Citizen'

df['AGE_GROUP'] = df['AGE'].apply(get_age_group)

# Plotting the proportion of applicants with "TARGET"=1 within each age group
prop_target_by_age_group = df[df['TARGET'] == 1].groupby('AGE_GROUP')['TARGET'].count() / df.groupby('AGE_GROUP')['TARGET'].count()
prop_target_by_age_group.plot(kind='bar')

plt.show()

# Observations
# senior citizens<middle aged<young<very young. Very young only went up to about .13 though (senior citizen was just under .05)


mask = (df['AGE_GROUP'] == 'Very_Young') & (df['CODE_GENDER'] == 'M') & (df['TARGET'] == 1)
fask = (df['AGE_GROUP'] == 'Very_Young') & (df['CODE_GENDER'] == 'F') & (df['TARGET'] == 1)
df_filtered = df[mask]
df_Ffiltered = df[fask]

print(len(df_filtered))
print(len(df_Ffiltered))

target_count = df['TARGET'].sum()
print(target_count)

mask = (df['AGE_GROUP'] == 'Very_Young') & (df['TARGET'] == 1)
vyoung_target_count = len(df[mask])

print(vyoung_target_count)

# Observations
#
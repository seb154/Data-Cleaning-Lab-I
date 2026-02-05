
# %% [markdown]
# # Step 1: Questions

# %% [markdown]
# College Completion Dataset: Is there a correlation between estimated
#  median SAT score and student completion/graduation rate?

# %% [markdown]
# Job Placement Dataset: Which degree type is compensaited the highest
#  by corporate to candidats?

# %% [markdown]
# # Step 2: 

# %% [markdown]
# Independent Business Metric for College Completion Dataset: Estimated median SAT score and student completion/graduation rate (%)
# %% [markdown]
# Independent Business Metric for Job Placement Dataset: Average salary/compensation for each degree type

# %% [markdown]
# # College Completion Dataset Preparation:

# %% [markdown]
# Correct Variable Types; Collapse factor levels as necessary
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
cc_df = pd.read_csv("cc_institution_details.csv")
cc_df.info()

#Correct Variable Types:
Column_index_list = [6,8,9]
cc_df.iloc[:,Column_index_list]= cc_df.iloc[:,Column_index_list].astype('category') 
#print(cc_df.iloc[:, 6])

#Collapse factor levels as necessary:
cc_df.control.value_counts() #looks good
cc_df.hbcu.value_counts() #looks good
cc_df.flagship.value_counts() #looks good


# %% [markdown]
# Normailize numerical variables as necessary
# %%
#Normailize numerical variables as necessary:
from sklearn.preprocessing import MinMaxScaler

cc_df.awards_per_value.plot.density() #example before normalization

num_cols = list(cc_df.select_dtypes('number'))
cc_df[num_cols] = MinMaxScaler().fit_transform(cc_df[num_cols])

cc_df.awards_per_value.plot.density() #relationships are the same after example normalization

# %% [markdown]
# One Hot Encode categorical variables as necessary
# %%
#One Hot Encode categorical variables as necessary:
category_list = list(cc_df.select_dtypes('category'))
cc_df_1h = pd.get_dummies(cc_df, columns = category_list) 
cc_df_1h.head()

# %% [markdown]
# Target Variable Creation; Prevalence of Target Variable
# %%
#Target Variable Creation:
cutoff = cc_df_1h['vsa_grad_elsewhere_after6_first'].quantile(0.75)
cc_df_1h['vsa_grad_elsewhere_after6_first_f'] = (cc_df_1h['vsa_grad_elsewhere_after6_first'] >= cutoff).astype(int)
 
#Prevalence of Target Variable:
prevalence_college = cc_df_1h['vsa_grad_elsewhere_after6_first_f'].value_counts()[1] / len(cc_df_1h['vsa_grad_elsewhere_after6_first_f'])

# %% [markdown]: 
# Train,Tune,Test Split
# %%
from sklearn.model_selection import train_test_split 
Train_c, Test_c = train_test_split(
    cc_df_1h,
    train_size=0.7,
    stratify=cc_df_1h.vsa_grad_elsewhere_after6_first_f
)

Tune_c, Test_c = train_test_split(
    Test_c,
    train_size=0.5,
    stratify=Test_c.vsa_grad_elsewhere_after6_first_f
)
print(Train_c.shape, Tune_c.shape, Test_c.shape)

# %% [markdown]
# # Job Placement Dataset Preparation:

# %% [markdown]
# Correct Variable Types; Collapse factor levels as necessary
# %%
job_df = pd.read_csv("Placement_Data_Full_Class.csv")
job_df.info()

#Correct Variable Types:
Column_index_list = [1,8,9,11,13]
job_df.iloc[:,Column_index_list]= job_df.iloc[:,Column_index_list].astype('category') 


#Collapse factor levels as necessary:
job_df.gender.value_counts() #looks good
job_df.degree_t.value_counts() #looks good
job_df.workex.value_counts() #looks good
job_df.specialisation.value_counts() #looks good
job_df.status.value_counts() #looks good

# %% [markdown]
# Normailize numerical variables as necessary
# %%
#Normailize numerical variables as necessary:
num_cols = list(job_df.select_dtypes('number'))
job_df[num_cols] = MinMaxScaler().fit_transform(job_df[num_cols])

job_df.degree_p.plot.density() #example after normalization


# %% [markdown]
# One Hot Encode categorical variables as necessary
# %%
#One Hot Encode categorical variables as necessary:
category_list = list(job_df.select_dtypes('category'))
job_df_1h = pd.get_dummies(job_df, columns = category_list) 
job_df_1h.head()

# %% [markdown]
# Target Variable Creation; Prevalence of Target Variable
# %%
#Target Variable Creation:
cut = job_df_1h['salary'].quantile(0.75)
job_df_1h['salary_f'] = (job_df_1h['salary'] >= cut).astype(int)
 
#Prevalence of Target Variable:
prevalence_job = job_df_1h['salary_f'].value_counts()[1] / len(job_df_1h['salary_f'])

# %% [markdown]: 
# Train,Tune,Test Split
# %%
Train_c, Test_c = train_test_split(
    job_df_1h,
    train_size=0.7,
    stratify=job_df_1h.salary_f
)

Tune_c, Test_c = train_test_split(
    Test_c,
    train_size=0.5,
    stratify=Test_c.salary_f
)
print(Train_c.shape, Tune_c.shape, Test_c.shape)

# %% [markdown]
# # Step 3: 
# The data can adress part of the question, but not all of it. The question at hand was broad and
#  not specific, the data didn't have the capacity to answer the question in its entirety. 
# Also, the data can only show correlation, not causation.

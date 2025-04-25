import pandas as pd #type: ignore
import os #type: ignore
import numpy as np #type: ignore
import random
import math

fp = os.path.expanduser('~/code_analysis/data/modeld_succ_combined_0519-25.csv')
df = pd.read_csv(fp)

grouped = df.groupby(['ScenarioName', 'InteractionId'])

request_counts = grouped.size().reset_index(name='RequestCount')

request_count_stats = request_counts.groupby('ScenarioName')['RequestCount'].agg(['mean',
                                                                                  lambda x: x.quantile(0.05),
                                                                                  lambda x: x.quantile(0.10),
                                                                                  lambda x: x.quantile(0.50),
                                                                                  lambda x: x.quantile(0.75),
                                                                                  lambda x: x.quantile(0.90),
                                                                                  lambda x: x.quantile(0.99)])
request_count_stats.columns = ['Mean', '5th Percentile', '10th Percentile', '50th Percentile', '75th Percentile', '90th Percentile', '99th Percentile']

request_count_stats = request_count_stats.sort_values(by='99th Percentile', ascending=False)

request_count_stats.reset_index().to_csv('~/code_analysis/data/llmcall_count_app_stats_0519-25.csv', index=False)

# process requests

fp1 = os.path.expanduser('~/code_analysis/data/modeldqaseus2db_cogsvc_052224_all.csv')
fp2 = os.path.expanduser('~/code_analysis/data/modeldqaseus2db_cogsvc_052224_cached.csv')
fp3 = os.path.expanduser('~/code_analysis/sec3.4/modelcogcomb.csv')

df1 = pd.read_csv(fp1)
df2 = pd.read_csv(fp2)
df3 = pd.read_csv(fp3)

grouped1 = df1.groupby('ScenarioName').agg({
    'contexttokennum': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 99)],
    'generatedtokennum': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 99)]
}).reset_index()

grouped2 = df2.groupby('ScenarioName').agg({
    'cachedTokens': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 99)]
}).reset_index()

grouped3 = df3.groupby('ScenarioName').agg({
    'contexttokennum': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 99)],
    'generatedtokennum': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 99)],
    'cachedtokennum': ['mean', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 99)]
}).reset_index()


grouped1.columns = ['ScenarioName', 'context_mean', 'context_10th', 'context_99th', 
                   'generated_mean', 'generated_10th', 'generated_99th']

grouped2.columns = ['ScenarioName', 'cached_mean', 'cached_10th', 'cached_99th']

grouped3.columns = ['ScenarioName', 'context_mean', 'context_10th', 'context_99th', 
                   'generated_mean', 'generated_10th', 'generated_99th', 'cached_mean', 'cached_10th', 'cached_99th']


df1 = df1.merge(grouped1[['ScenarioName', 'context_mean', 'context_99th', 'generated_mean', 'generated_99th']], on='ScenarioName', how='left')
df2 = df2.merge(grouped2[['ScenarioName', 'cached_mean', 'cached_99th']], on='ScenarioName', how='left')
df3 = df3.merge(grouped3[['ScenarioName', 'context_mean','context_99th', 'generated_mean', 'generated_99th', 'cached_mean','cached_99th']], on='ScenarioName', how='left')

df1 = df1.merge(grouped2[['ScenarioName', 'cached_mean', 'cached_99th']], on='ScenarioName', how='left')
df1['cached_99th'] = df1['cached_99th'].fillna(df1['context_99th'] * 0.1)
df1['cached_mean'] = df1['cached_mean'].fillna(df1['context_mean'] * 0.1)
df3['cached_99th'] = df3['cached_99th'].fillna(df3['context_99th'] * 0.1)
df3['cached_mean'] = df3['cached_mean'].fillna(df3['context_mean'] * 0.1)

df1 = df1.rename(columns={'context_99th': 'input99app'})
df1 = df1.rename(columns={'context_mean': 'inputmeanapp'})
df1 = df1.rename(columns={'generated_99th': 'output99app'})
df1 = df1.rename(columns={'generated_mean': 'outputmeanapp'})
df1 = df1.rename(columns={'cached_99th': 'sys99app'})
df1 = df1.rename(columns={'cached_mean': 'sysmeanapp'})
df1 = df1.rename(columns={'contexttokennum': 'prompt_len'})
df1 = df1.rename(columns={'generatedtokennum': 'output_len'})

df3 = df3.rename(columns={'context_99th': 'input99app'})
df3 = df3.rename(columns={'context_mean': 'inputmeanapp'})
df3 = df3.rename(columns={'generated_99th': 'output99app'})
df3 = df3.rename(columns={'generated_mean': 'outputmeanapp'})
df3 = df3.rename(columns={'cached_99th': 'sys99app'})
df3 = df3.rename(columns={'cached_mean': 'sysmeanapp'})
df3 = df3.rename(columns={'cachedtokennum': 'syslen'})
df3 = df3.rename(columns={'contexttokennum': 'prompt_len'})
df3 = df3.rename(columns={'generatedtokennum': 'output_len'})

percentages = [0.1, 0.2, 0.3, 0.4, 0.5]

df1['sys99app'] = df1.apply(lambda row: row['input99app'] * 0.1 if row['sys99app'] == 0 else row['sys99app'], axis=1)
df1['sysmeanapp'] = df1.apply(lambda row: row['inputmeanapp'] * 0.1 if row['sysmeanapp'] == 0 else row['sysmeanapp'], axis=1)
df3['sys99app'] = df3.apply(lambda row: row['input99app'] * 0.1 if row['sys99app'] == 0 else row['sys99app'], axis=1)
df3['sysmeanapp'] = df3.apply(lambda row: row['inputmeanapp'] * 0.1 if row['sysmeanapp'] == 0 else row['sysmeanapp'], axis=1)

df1['inputsysratio'] = df1['input99app']/df1['sys99app']
df3['inputsysratio'] = df3['input99app']/df3['sys99app']

def calculate_syslen(row):
    if row['inputsysratio'] < 1:
        return row['inputsysratio'] * row['prompt_len']
    else:
        return random.choice([0.1, 0.2, 0.3, 0.4, 0.5]) * row['prompt_len']

# df1['syslen'] = df1['prompt_len'].apply(lambda x: int(x * random.choice(percentages)))
# df3['syslen'] = df3['prompt_len'].apply(lambda x: int(x * random.choice(percentages)))

df1['syslen'] = df1.apply(calculate_syslen, axis=1)
df3['syslen'] = df3.apply(calculate_syslen, axis=1)

llmcall_fp = os.path.expanduser('~/code_analysis/data/llmcall_count_app_stats_0519-25.csv')
llmcall_stats = pd.read_csv(llmcall_fp)

merged_df = pd.merge(df1, llmcall_stats[['ScenarioName', 'Mean']], on='ScenarioName', how='left')
merged_df3 = pd.merge(df3, llmcall_stats[['ScenarioName', 'Mean']], on='ScenarioName', how='left')

# print(f"before merged_df: {len(merged_df)}")
# merged_df = merged_df.dropna(subset=['Mean'])
# print(f"after merged_df: {len(merged_df)}")

merged_df['Mean'] = merged_df['Mean'].fillna(1)
merged_df['llmcalls'] = merged_df['Mean'].apply(lambda x: math.ceil(x))

# print(f"before merged_df3: {len(merged_df3)}")
# merged_df3 = merged_df3.dropna(subset=['Mean'])
# print(f"after merged_df3: {len(merged_df3)}")

merged_df3['Mean'] = merged_df3['Mean'].fillna(1)
merged_df3['llmcalls'] = merged_df3['Mean'].apply(lambda x: math.ceil(x))

merged_df['RequestId'] = range(1, len(merged_df) + 1)
merged_df3['RequestId'] = range(1, len(merged_df3) + 1)

scenario_mapping = {name: idx for idx, name in enumerate(merged_df['ScenarioName'].unique(), 1)}
scenario_mapping_df3 = {name: idx for idx, name in enumerate(merged_df3['ScenarioName'].unique(), 1)}

merged_df['ScenarioId'] = merged_df['ScenarioName'].map(scenario_mapping)
merged_df3['ScenarioId'] = merged_df3['ScenarioName'].map(scenario_mapping_df3)

merged_df = merged_df.rename(columns={'ScenarioId': 'app'})
merged_df3 = merged_df3.rename(columns={'ScenarioId': 'app'})

#print(merged_df.head())
merged_df = merged_df.loc[:, ['RequestId', 'ScenarioName', 'prompt_len', 'output_len', 'app', 'inputmeanapp','input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']]
merged_df3 = merged_df3.loc[:, ['RequestId', 'ScenarioName', 'prompt_len', 'output_len', 'app', 'inputmeanapp','input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']]

max_input99app_df = merged_df['input99app'].max()
max_sys99app_df = merged_df['sys99app'].max()
max_output99app_df = merged_df['output99app'].max()

max_input99app_df3 = merged_df3['input99app'].max()
max_sys99app_df3 = merged_df3['sys99app'].max()
max_output99app_df3 = merged_df3['output99app'].max()

merged_df['prompt_len'] = (merged_df['prompt_len'] / max_input99app_df) * 2000
merged_df['input99app'] = (merged_df['input99app'] / max_input99app_df) * 2000
merged_df['inputmeanapp'] = (merged_df['inputmeanapp'] / max_input99app_df) * 2000
merged_df['syslen'] = (merged_df['syslen'] / max_sys99app_df) * 2000
merged_df['sys99app'] = (merged_df['sys99app'] / max_sys99app_df) * 2000
merged_df['sysmeanapp'] = (merged_df['sysmeanapp'] / max_sys99app_df) * 2000
merged_df['output_len'] = (merged_df['output_len'] / max_output99app_df) * 2000
merged_df['output99app'] = (merged_df['output99app'] / max_output99app_df) * 2000
merged_df['outputmeanapp'] = (merged_df['outputmeanapp'] / max_output99app_df) * 2000

merged_df3['prompt_len'] = (merged_df3['prompt_len'] / max_input99app_df3) * 2000
merged_df3['input99app'] = (merged_df3['input99app'] / max_input99app_df3) * 2000
merged_df3['inputmeanapp'] = (merged_df3['inputmeanapp'] / max_input99app_df3) * 2000
merged_df3['syslen'] = (merged_df3['syslen'] / max_sys99app_df3) * 2000
merged_df3['sys99app'] = (merged_df3['sys99app'] / max_sys99app_df3) * 2000
merged_df3['sysmeanapp'] = (merged_df3['sysmeanapp'] / max_sys99app_df3) * 2000
merged_df3['output_len'] = (merged_df3['output_len'] / max_output99app_df3) * 2000
merged_df3['output99app'] = (merged_df3['output99app'] / max_output99app_df3) * 2000
merged_df3['outputmeanapp'] = (merged_df3['outputmeanapp'] / max_output99app_df3) * 2000

merged_df[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp' ,'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']] = merged_df[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp' ,'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']].astype(int)
merged_df3[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp' ,'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']] = merged_df3[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp' ,'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']].astype(int)

merged_df = merged_df[(merged_df != 0).all(axis=1)]
merged_df3 = merged_df3[(merged_df3 != 0).all(axis=1)]

merged_df = merged_df.dropna()
merged_df3 = merged_df3.dropna()

merged_df = merged_df.sort_values(by='app')
merged_df3 = merged_df3.sort_values(by='app')

app_counts = merged_df['app'].value_counts()
app_counts3 = merged_df3['app'].value_counts()

# print(f"df app counts: {app_counts}")
# print(f"df3 app counts: {app_counts3}")
#print(f"before remove: {len(merged_df)}")
#merged_df = merged_df[(merged_df['prompt_len'] < merged_df['input99app']) & (merged_df['output_len'] < merged_df['output99app']) & (merged_df['syslen'] < merged_df['sys99app'])]
#print(f"after remove: {len(merged_df)}")
#merged_df3 = merged_df3[(merged_df3['prompt_len'] < merged_df3['input99app']) & (merged_df3['output_len'] < merged_df3['output99app']) & (merged_df3['syslen'] < merged_df3['sys99app'])]

merged_df['RequestId'] = range(1, len(merged_df) + 1)
merged_df3['RequestId'] = range(1, len(merged_df3) + 1)

merged_df_anon = merged_df.loc[:, ['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp' , 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']]
merged_df3_anon = merged_df3.loc[:, ['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp' , 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']]

hd_10k = os.path.expanduser('~/sampled_10k.csv') #home directory 10k path
hd_10k_df3 = os.path.expanduser('~/sampled_10k_df3.csv')

hd_10k_anon = os.path.expanduser('~/anon_sampled_10k.csv') #home directory 10k path
hd_10k_df3_anon = os.path.expanduser('~/anon_sampled_10k_df3.csv')

merged_df.to_csv(hd_10k, index=False)
merged_df3.to_csv(hd_10k_df3, index=False)

merged_df_anon.to_csv(hd_10k_anon, index=False)
merged_df3_anon.to_csv(hd_10k_df3_anon, index=False)

print(f"length of 1 day traces: {len(merged_df_anon)} and {len(merged_df3_anon)}")
print("1 day trace processing complete.")

##########PROCESS LARGER FILE######################
df_10k_path = os.path.expanduser('~/sampled_10k_df3.csv')

df_10k = pd.read_csv(df_10k_path)

unique_scenarios = df_10k.groupby('ScenarioName').agg({
    'app': 'first', 
    'inputmeanapp': 'first',         
    'input99app': 'first',
    'outputmeanapp': 'first',    
    'output99app': 'first',
    'sysmeanapp': 'first',   
    'sys99app': 'first',     
    'llmcalls': 'first'    
}).reset_index()

appmappath = os.path.expanduser('~/appmapping.csv')
unique_scenarios.to_csv(appmappath, index=False)

unique_scenarios_df = pd.read_csv(appmappath)
modeld_path = os.path.expanduser('~/code_analysis/data/modeld_succ_combined_0519-25.csv')
modeld_df = pd.read_csv(modeld_path)

merged_df_big = modeld_df.merge(unique_scenarios_df, on='ScenarioName', how='left')

#print(f"before operations len: {len(merged_df_big)}")
merged_df_big = merged_df_big.dropna(subset=['app', 'inputmeanapp','input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls'])
#print(f"after operations len: {len(merged_df_big)}")

merged_df_big['RequestId'] = range(1, len(merged_df_big) + 1)

merged_df_big['inputsysratio'] = merged_df_big['input99app']/merged_df_big['sys99app']

percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

merged_df_big['prompt_len'] = merged_df_big['input99app'].apply(lambda x: int(x * random.choice(percentages)))
merged_df_big['output_len'] = merged_df_big['output99app'].apply(lambda x: int(x * random.choice(percentages)))
merged_df_big['syslen'] = merged_df_big.apply(calculate_syslen, axis=1)
#merged_df_big['syslen'] = merged_df_big['sys99app'].apply(lambda x: int(x * random.choice(percentages)))


result_df = merged_df_big.loc[:, ['RequestId', 'ScenarioName', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']]

result_df_anon = merged_df_big.loc[:, ['RequestId', 'ScenarioName', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']]


result_df[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']] = result_df[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']].astype(int)
result_df_anon[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']] = result_df_anon[['RequestId', 'prompt_len', 'output_len', 'app', 'inputmeanapp', 'input99app', 'outputmeanapp', 'output99app', 'sysmeanapp', 'sys99app', 'llmcalls', 'syslen']].astype(int)

result_df = result_df[(result_df != 0).all(axis=1)]
result_df_anon = result_df_anon[(result_df_anon != 0).all(axis=1)]

result_df = result_df.dropna()
result_df_anon = result_df_anon.dropna()

result_df = result_df.sort_values(by='app')
result_df_anon = result_df_anon.sort_values(by='app')

app_counts_full = result_df['app'].value_counts()
app_counts_full_anon = result_df_anon['app'].value_counts()

#print(f"app_counts_full: {app_counts_full}")

result_df['RequestId'] = range(1, len(result_df) + 1)
result_df_anon['RequestId'] = range(1, len(result_df_anon) + 1)

fulltrace_path = os.path.expanduser('~/fulltrace.csv')
fulltrace_anon_path = os.path.expanduser('~/anon_fulltrace.csv')

result_df.to_csv(fulltrace_path, index=False)
result_df_anon.to_csv(fulltrace_anon_path, index=False)

print(f"length of full trace: {len(result_df_anon)}")
print("Full trace 7 days processing complete.")

# output_df = df[['RequestId', 'InteractionId', 'ScenarioName', 'contexttokennum', 
#                 'generatedtokennum', 'cachedtokennum', 'input99app', 'sys99app', 'output99app']]

# output_path = '/mnt/data/updated_requests.csv'
# output_df.to_csv(output_path, index=False)

# output_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MaxAbsScaler

# Assuming the CSV file is already available, replace 'file_path' with the actual path
input1 = '../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___lshare_fair___10___2000.jsonl'
input2 = '../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_wsc___10___2000.jsonl'
input3 = '../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_odt_wsc_limit___10___2000.jsonl'
input4 = '../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit___10___2000.jsonl'
input5 = '../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_wsc_limit___10___2000.jsonl'

names = ["unnamed","RPM", "FS (W)", "FS (W+O)", "FS (W+O+I)", "FS (W+L)", "VTC"]

exps_1 = []
exps_2 = []
exps_3 = []
exps_4 = []
exps_5 = []
with open(input1, "r") as f:
    lines = f.readlines()
    for line in lines:
        exps_1.append({})
        exps_1[-1]["config"] = json.loads(line)["config"]
        exps_1[-1]["result"] = json.loads(line)["result"]

with open(input2, "r") as f:
    lines = f.readlines()
    for line in lines:
        exps_2.append({})
        exps_2[-1]["config"] = json.loads(line)["config"]
        exps_2[-1]["result"] = json.loads(line)["result"]

with open(input3, "r") as f:
    lines = f.readlines()
    for line in lines:
        exps_3.append({})
        exps_3[-1]["config"] = json.loads(line)["config"]
        exps_3[-1]["result"] = json.loads(line)["result"]

with open(input4, "r") as f:
    lines = f.readlines()
    for line in lines:
        exps_4.append({})
        exps_4[-1]["config"] = json.loads(line)["config"]
        exps_4[-1]["result"] = json.loads(line)["result"]

with open(input5, "r") as f:
    lines = f.readlines()
    for line in lines:
        exps_5.append({})
        exps_5[-1]["config"] = json.loads(line)["config"]
        exps_5[-1]["result"] = json.loads(line)["result"]


for exp in exps_1:
    config = exp["config"]
    result = exp["result"]
    responses = result["responses"]
    df1 = pd.DataFrame(responses)

for exp in exps_2:
    config = exp["config"]
    result = exp["result"]
    responses = result["responses"]
    df2 = pd.DataFrame(responses)

for exp in exps_3:
    config = exp["config"]
    result = exp["result"]
    responses = result["responses"]
    df3 = pd.DataFrame(responses)

for exp in exps_4:
    config = exp["config"]
    result = exp["result"]
    responses = result["responses"]
    df4 = pd.DataFrame(responses)

for exp in exps_5:
    config = exp["config"]
    result = exp["result"]
    responses = result["responses"]
    df5 = pd.DataFrame(responses)

#print("")
# Group by 'adapter_dir' and count the number of rows in each group
grouped_counts_df1 = df1.groupby('adapter_dir').size().reset_index(name='RPM')
df1_with_counts = pd.merge(df1, grouped_counts_df1, on='adapter_dir')

grouped_counts_df2 = df2.groupby('adapter_dir').size().reset_index(name='RPM')
df2_with_counts = pd.merge(df2, grouped_counts_df2, on='adapter_dir')

grouped_counts_df3 = df3.groupby('adapter_dir').size().reset_index(name='RPM')
df3_with_counts = pd.merge(df3, grouped_counts_df3, on='adapter_dir')

grouped_counts_df4 = df4.groupby('adapter_dir').size().reset_index(name='RPM')
df4_with_counts = pd.merge(df4, grouped_counts_df4, on='adapter_dir')

grouped_counts_df5 = df5.groupby('adapter_dir').size().reset_index(name='RPM')
df5_with_counts = pd.merge(df5, grouped_counts_df5, on='adapter_dir')


df1 = df1_with_counts
df2 = df2_with_counts
df3 = df3_with_counts
df4 = df4_with_counts
df5 = df5_with_counts

df1['service_ratio'] = (df1['prompt_len'] / df1['input99app']) + (df1['sys_len'] / df1['sys99app']) + 2 * (df1['output_len'] / df1['output99app'])
df2['service_ratio'] = (df2['prompt_len'] / df2['input99app']) + (df2['sys_len'] / df2['sys99app']) + 2 * (df2['output_len'] / df2['output99app'])
df3['service_ratio'] = (df3['prompt_len'] / df3['input99app']) + (df3['sys_len'] / df3['sys99app']) + 2 * (df3['output_len'] / df3['output99app'])
df4['service_ratio'] = (df4['prompt_len'] / df4['input99app']) + (df4['sys_len'] / df4['sys99app']) + 2 * (df4['output_len'] / df4['output99app'])
df5['service_ratio'] = (df5['prompt_len'] / df5['input99app']) + (df5['sys_len'] / df5['sys99app']) + 2 * (df5['output_len'] / df5['output99app'])

throttled_requests_df1 = df1[df1['first_token_latency'] == -1]
throttled_requests_df2 = df2[df2['first_token_latency'] == -1]
throttled_requests_df3 = df3[df3['first_token_latency'] == -1]
throttled_requests_df4 = df4[df4['first_token_latency'] == -1]
throttled_requests_df5 = df5[df5['first_token_latency'] == -1]
#print(len(throttled_requests_df1))

throttled_user_stats_df1 = throttled_requests_df1.groupby('adapter_dir')[['prompt_len', 'output_len', 'service_ratio', 'RPM']].mean().reset_index()
throttled_user_stats_df2 = throttled_requests_df2.groupby('adapter_dir')[['prompt_len', 'output_len', 'service_ratio', 'RPM']].mean().reset_index()
throttled_user_stats_df3 = throttled_requests_df3.groupby('adapter_dir')[['prompt_len', 'output_len', 'service_ratio', 'RPM']].mean().reset_index()
throttled_user_stats_df4 = throttled_requests_df4.groupby('adapter_dir')[['prompt_len', 'output_len', 'service_ratio', 'RPM']].mean().reset_index()
throttled_user_stats_df5 = throttled_requests_df5.groupby('adapter_dir')[['prompt_len', 'output_len', 'service_ratio', 'RPM']].mean().reset_index()

scaler = MaxAbsScaler()
if len(throttled_user_stats_df1) > 0:
    throttled_user_stats_df1[['prompt_len', 'output_len', 'service_ratio', 'RPM']] = scaler.fit_transform(throttled_user_stats_df1[['prompt_len', 'output_len', 'service_ratio', 'RPM']])
if len(throttled_user_stats_df2) > 0:
    throttled_user_stats_df2[['prompt_len', 'output_len', 'service_ratio', 'RPM']] = scaler.fit_transform(throttled_user_stats_df2[['prompt_len', 'output_len', 'service_ratio', 'RPM']])
if len(throttled_user_stats_df3) > 0:
    throttled_user_stats_df3[['prompt_len', 'output_len', 'service_ratio', 'RPM']] = scaler.fit_transform(throttled_user_stats_df3[['prompt_len', 'output_len', 'service_ratio', 'RPM']])
if len(throttled_user_stats_df4) > 0:
    throttled_user_stats_df4[['prompt_len', 'output_len', 'service_ratio', 'RPM']] = scaler.fit_transform(throttled_user_stats_df4[['prompt_len', 'output_len', 'service_ratio', 'RPM']])
if len(throttled_user_stats_df5) > 0:
    throttled_user_stats_df5[['prompt_len', 'output_len', 'service_ratio', 'RPM']] = scaler.fit_transform(throttled_user_stats_df5[['prompt_len', 'output_len', 'service_ratio', 'RPM']])

print(len(throttled_user_stats_df1))
print(len(throttled_user_stats_df2))
print(len(throttled_user_stats_df3))
print(len(throttled_user_stats_df4))
print(len(throttled_user_stats_df5))

sorted_prompt_len_df1 = np.sort(throttled_user_stats_df1['prompt_len'])
sorted_prompt_len_df2 = np.sort(throttled_user_stats_df2['prompt_len'])
sorted_prompt_len_df3 = np.sort(throttled_user_stats_df3['prompt_len'])
sorted_prompt_len_df4 = np.sort(throttled_user_stats_df4['prompt_len'])
sorted_prompt_len_df5 = np.sort(throttled_user_stats_df5['prompt_len'])

sorted_prompt_len_df1 = np.insert(sorted_prompt_len_df1, 0, 0)
sorted_prompt_len_df2 = np.insert(sorted_prompt_len_df2, 0, 0)
sorted_prompt_len_df3 = np.insert(sorted_prompt_len_df3, 0, 0)
sorted_prompt_len_df4 = np.insert(sorted_prompt_len_df4, 0, 0)
sorted_prompt_len_df5 = np.insert(sorted_prompt_len_df5, 0, 0)

#print(len(sorted_prompt_len))
sorted_output_len_df1 = np.sort(throttled_user_stats_df1['output_len'])
sorted_output_len_df2 = np.sort(throttled_user_stats_df2['output_len'])
sorted_output_len_df3 = np.sort(throttled_user_stats_df3['output_len'])
sorted_output_len_df4 = np.sort(throttled_user_stats_df4['output_len'])
sorted_output_len_df5 = np.sort(throttled_user_stats_df5['output_len'])

sorted_output_len_df1 = np.insert(sorted_output_len_df1, 0, 0)
sorted_output_len_df2 = np.insert(sorted_output_len_df2, 0, 0)
sorted_output_len_df3 = np.insert(sorted_output_len_df3, 0, 0)
sorted_output_len_df4 = np.insert(sorted_output_len_df4, 0, 0)
sorted_output_len_df5 = np.insert(sorted_output_len_df5, 0, 0)

#print(len(sorted_prompt_len))
sorted_service_ratio_df1 = np.sort(throttled_user_stats_df1['service_ratio'])
sorted_service_ratio_df2 = np.sort(throttled_user_stats_df2['service_ratio'])
sorted_service_ratio_df3 = np.sort(throttled_user_stats_df3['service_ratio'])
sorted_service_ratio_df4 = np.sort(throttled_user_stats_df4['service_ratio'])
sorted_service_ratio_df5 = np.sort(throttled_user_stats_df5['service_ratio'])

sorted_service_ratio_df1 = np.insert(sorted_service_ratio_df1, 0, 0)
sorted_service_ratio_df2 = np.insert(sorted_service_ratio_df2, 0, 0)
sorted_service_ratio_df3 = np.insert(sorted_service_ratio_df3, 0, 0)
sorted_service_ratio_df4 = np.insert(sorted_service_ratio_df4, 0, 0)
sorted_service_ratio_df5 = np.insert(sorted_service_ratio_df5, 0, 0)

sorted_rpm_df1 = np.sort(throttled_user_stats_df1['RPM'])
sorted_rpm_df2 = np.sort(throttled_user_stats_df2['RPM'])
sorted_rpm_df3 = np.sort(throttled_user_stats_df3['RPM'])
sorted_rpm_df4 = np.sort(throttled_user_stats_df4['RPM'])
sorted_rpm_df5 = np.sort(throttled_user_stats_df5['RPM'])

sorted_rpm_df1 = np.insert(sorted_rpm_df1, 0, 0)
sorted_rpm_df2 = np.insert(sorted_rpm_df2, 0, 0)
sorted_rpm_df3 = np.insert(sorted_rpm_df3, 0, 0)
sorted_rpm_df4 = np.insert(sorted_rpm_df4, 0, 0)
sorted_rpm_df5 = np.insert(sorted_rpm_df5, 0, 0)

cdf_prompt_len_df1 = np.arange(1, len(sorted_prompt_len_df1) + 1) / len(sorted_prompt_len_df1)
cdf_output_len_df1 = np.arange(1, len(sorted_output_len_df1) + 1) / len(sorted_output_len_df1)
cdf_service_ratio_df1 = np.arange(1, len(sorted_service_ratio_df1) + 1) / len(sorted_service_ratio_df1)
cdf_rpm_df1 = np.arange(1, len(sorted_rpm_df1) + 1) / len(sorted_rpm_df1)

cdf_prompt_len_df2 = np.arange(1, len(sorted_prompt_len_df2) + 1) / len(sorted_prompt_len_df2)
cdf_output_len_df2 = np.arange(1, len(sorted_output_len_df2) + 1) / len(sorted_output_len_df2)
cdf_service_ratio_df2 = np.arange(1, len(sorted_service_ratio_df2) + 1) / len(sorted_service_ratio_df2)
cdf_rpm_df2 = np.arange(1, len(sorted_rpm_df2) + 1) / len(sorted_rpm_df2)

cdf_prompt_len_df3 = np.arange(1, len(sorted_prompt_len_df3) + 1) / len(sorted_prompt_len_df3)
cdf_output_len_df3 = np.arange(1, len(sorted_output_len_df3) + 1) / len(sorted_output_len_df3)
cdf_service_ratio_df3 = np.arange(1, len(sorted_service_ratio_df3) + 1) / len(sorted_service_ratio_df3)
cdf_rpm_df3 = np.arange(1, len(sorted_rpm_df3) + 1) / len(sorted_rpm_df3)

cdf_prompt_len_df4 = np.arange(1, len(sorted_prompt_len_df4) + 1) / len(sorted_prompt_len_df4)
cdf_output_len_df4 = np.arange(1, len(sorted_output_len_df4) + 1) / len(sorted_output_len_df4)
cdf_service_ratio_df4 = np.arange(1, len(sorted_service_ratio_df4) + 1) / len(sorted_service_ratio_df4)
cdf_rpm_df4 = np.arange(1, len(sorted_rpm_df4) + 1) / len(sorted_rpm_df4)

cdf_prompt_len_df5 = np.arange(1, len(sorted_prompt_len_df5) + 1) / len(sorted_prompt_len_df5)
cdf_output_len_df5 = np.arange(1, len(sorted_output_len_df5) + 1) / len(sorted_output_len_df5)
cdf_service_ratio_df5 = np.arange(1, len(sorted_service_ratio_df5) + 1) / len(sorted_service_ratio_df5)
cdf_rpm_df5 = np.arange(1, len(sorted_rpm_df5) + 1) / len(sorted_rpm_df5)

# Plot the Cdf for prompt_len and output_len

# print(sorted_prompt_len_df2)
# print(cdf_prompt_len_df2)
sorted_prompt_len_df2 = np.array([0,1])
cdf_prompt_len_df2 = np.array([0,0]) 
#cdf_prompt_len_df2 = np.arange(1, len(sorted_prompt_len_df2) + 1) / len(sorted_prompt_len_df2)
# print(sorted_prompt_len_df1)
# print(cdf_prompt_len_df1)

print(sorted_prompt_len_df3)
print(cdf_prompt_len_df3)

plt.figure(figsize=(10, 6))
plt.plot(sorted_prompt_len_df1, cdf_prompt_len_df1, label=names[1], color='violet')
plt.plot(sorted_prompt_len_df2, cdf_prompt_len_df2, label=names[2], color='blue')
plt.plot(sorted_prompt_len_df3, cdf_prompt_len_df3, label=names[3], color='green')
plt.plot(sorted_prompt_len_df4, cdf_prompt_len_df4, label=names[4], color='orange')
plt.plot(sorted_prompt_len_df5, cdf_prompt_len_df5, label=names[5], color='red')
plt.xlabel('Normalized input length')
plt.ylabel('CDF of throttled users')
plt.legend()
plt.grid(True)
plt.savefig("cdf_throttleuservsinput.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
plt.plot(sorted_output_len_df1, cdf_output_len_df1, label=names[1], color='violet')
plt.plot(sorted_output_len_df2, cdf_output_len_df2, label=names[2], color='blue')
plt.plot(sorted_output_len_df3, cdf_output_len_df3, label=names[3], color='green')
plt.plot(sorted_output_len_df4, cdf_output_len_df4, label=names[4], color='orange')
plt.plot(sorted_output_len_df5, cdf_output_len_df5, label=names[5], color='red')
plt.xlabel('Normalized output length')
plt.ylabel('CDF of throttled users')
plt.legend()
plt.grid(True)
plt.savefig("cdf_throttleuservsoutput.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
plt.plot(sorted_service_ratio_df1, cdf_service_ratio_df1, label=names[1], color='violet')
plt.plot(sorted_service_ratio_df2, cdf_service_ratio_df2, label=names[2], color='blue')
plt.plot(sorted_service_ratio_df3, cdf_service_ratio_df3, label=names[3], color='green')
plt.plot(sorted_service_ratio_df4, cdf_service_ratio_df4, label=names[4], color='orange')
plt.plot(sorted_service_ratio_df5, cdf_service_ratio_df5, label=names[5], color='red')
plt.xlabel('Normalized weighted service')
plt.ylabel('CDF of throttled users')
plt.legend()
plt.grid(True)
plt.savefig("cdf_throttleuservsserviceratio.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
plt.plot(sorted_rpm_df1, cdf_rpm_df1, label=names[1], color='violet')
plt.plot(sorted_rpm_df2, cdf_rpm_df2, label=names[2], color='blue')
plt.plot(sorted_rpm_df3, cdf_rpm_df3, label=names[3], color='green')
plt.plot(sorted_rpm_df4, cdf_rpm_df4, label=names[4], color='orange')
plt.plot(sorted_rpm_df5, cdf_rpm_df5, label=names[5], color='red')
plt.xlabel('Normalized RPM')
plt.ylabel('CDF of throttled users')
plt.legend()
plt.grid(True)
plt.savefig("cdf_throttleuservsrpm.pdf", bbox_inches="tight")
# # Cdf for prompt_len
# plt.plot(sorted_prompt_len_df1, cdf_prompt_len_df1, label='Prompt Length', color='b')

# # Cdf for output_len
# plt.plot(sorted_output_len_df1, cdf_output_len_df1, label='Output Length', color='r')

# plt.plot(sorted_service_ratio_df1, cdf_service_ratio_df1, label='Service Ratio', color='green')

# plt.plot(sorted_rpm_df1, cdf_rpm_df1, label='RPM', color='violet')

#plt.title('Cdf of Throttled Users: Prompt Length vs Output Length')
# plt.xlabel('Length')
# plt.ylabel('Cdf of throttled users')
# plt.legend()
# plt.grid(True)
# plt.savefig("cdf_throttleuservsinputoutput.pdf", bbox_inches="tight")

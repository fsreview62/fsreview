import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Assuming the CSV is already present at a specific location, load the CSV into a DataFrame
#file_path = '../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit___50___2000.csv'
file_path = '../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___lshare_fair___50___2000.csv'
df = pd.read_csv(file_path)

# Calculate the new required columns
df['throughput'] = df['output_len'] / (df['request_latency'] - df['first_token_latency'])
df['service_received'] = ((df['prompt_len'] / df['input99app']) + 
                          (df['sys_len'] / df['sys99app']) + 
                          2 * (df['output_len'] / df['output99app'])) / (df['request_latency'] - df['first_token_latency'])

df['inputratio'] = df['prompt_len'] / df['input99app']
df['sysratio'] = df['sys_len'] / df['sys99app']
df['outratio'] = df['output_len'] / df['output99app']
df['service_ratio'] = (df['prompt_len'] / df['input99app']) + (df['sys_len'] / df['sys99app']) + 2 * (df['output_len'] / df['output99app'])
df['tokenpersec'] = (df['prompt_len'] + df['sys_len'] + 2 * df['output_len']) / (df['request_latency'] - df['first_token_latency'])

# Save the updated dataframe into a new CSV file
#updated_file_path = 'downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit___50___2000.csv_updated_requests.csv'
updated_file_path = 'downsampled_data_10000_rows_first_arrivals.csv___lshare_fair___50___2000.csv_updated_requests.csv'
df.to_csv(updated_file_path, index=False)

# Group by 'adapter_dir' and calculate the averages
grouped_df = df.groupby('adapter_dir').agg(
    avg_throughput=('throughput', 'mean'),
    avg_service=('service_received', 'mean'),
    avg_response=('first_token_latency', 'mean'),
    avg_serviceratio=('service_ratio', 'mean'),
    avg_tokenpersec=('tokenpersec', 'mean')
).reset_index()

# Merge the grouped values back to the original dataframe
df = df.merge(grouped_df, on='adapter_dir', how='left')

# Save the merged dataframe into a new CSV file
#grouped_file_path = 'downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit___50___2000.csv_updated_requests.csv_grouped_requests.csv'
grouped_file_path = 'downsampled_data_10000_rows_first_arrivals.csv___lshare_fair___50___2000.csv_updated_requests.csv_grouped_requests.csv'
df.to_csv(grouped_file_path, index=False)

# Load CSV into a dataframe (assuming file path is provided)
# file_path = 'downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit___50___2000.csv_updated_requests.csv_grouped_requests.csv'  # Example file path
# df = pd.read_csv(file_path)

# Assign each unique 'adapter_dir' (user) a unique id starting from 1
df['user_id'] = df.groupby('adapter_dir').ngroup() + 1

# Select the relevant columns for each user
user_metrics = df[['user_id', 'adapter_dir', 'service_ratio', 'avg_throughput', 'avg_service', 'avg_response', 'avg_serviceratio', 'avg_tokenpersec']]

# No need to calculate mean, just extract the first instance of each unique user (adapter_dir)
user_aggregates = user_metrics.drop_duplicates(subset=['user_id'])

max_serviceratio_user = user_aggregates.loc[user_aggregates['avg_throughput'].idxmax()]

print(max_serviceratio_user)
# # Find the user with the maximum avg_serviceratio
# max_serviceratio_user = user_aggregates.loc[user_aggregates['avg_serviceratio'].idxmax()]

# print(max_serviceratio_user)
# # Calculate the ratio of avg_serviceratio and avg_tokenpersec of max_serviceratio_user compared to other users
# user_aggregates['serviceratio_ratio'] = user_aggregates['avg_serviceratio'] / max_serviceratio_user['avg_serviceratio']
# user_aggregates['tokenpersec_ratio'] = user_aggregates['avg_tokenpersec'] / max_serviceratio_user['avg_tokenpersec']

# # Plot the results
# plt.figure(figsize=(10, 6))

# # Line plot for avg_serviceratio ratio
# plt.plot(user_aggregates['user_id'], user_aggregates['serviceratio_ratio'], label='Serviceratio Ratio', marker='o')

# # Line plot for avg_tokenpersec ratio
# plt.plot(user_aggregates['user_id'], user_aggregates['tokenpersec_ratio'], label='Tokenpersec Ratio', marker='x')

# plt.title('Comparison of avg_serviceratio and avg_tokenpersec with Max Serviceratio User')
# plt.xlabel('User ID')
# plt.ylabel('Ratio')
# plt.legend()
# plt.grid(True)
# plt.savefig("serviceacctonature.pdf", bbox_inches="tight")
# plt.show()

# Sort avg_throughput to calculate CDF
sorted_throughput = np.sort(user_aggregates['avg_throughput'])

# Calculate the cumulative probabilities
cdf = np.arange(1, len(sorted_throughput) + 1) / len(sorted_throughput)

# Plot CDF
plt.figure(figsize=(8, 6))
plt.plot(sorted_throughput, cdf, linestyle='-', color='b')
#plt.title('CDF of Avg Throughput')
plt.xlabel('Avg Throughput (tokens/sec)')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.savefig("cdfthroughput_lshare_fair.pdf", bbox_inches="tight")
import re
import matplotlib.pyplot as plt
from collections import defaultdict

input_file = 'out'  
collected_lines_file = 'collected_lines.txt'

# Regular expression to match the log line and extract the dictionaries
pattern = r"finished batch\. requests: (.*), input tokens per user: (.*), output tokens per user: (.*)"

requests_data = defaultdict(list)
input_tokens_data = defaultdict(list)
output_tokens_data = defaultdict(list)

with open(input_file, 'r') as infile, open(collected_lines_file, 'w') as outfile:
    for line in infile:
        match = re.match(pattern, line)
        if match:
            outfile.write(line) 

            requests = eval(match.group(1))
            input_tokens = eval(match.group(2))
            output_tokens = eval(match.group(3))
            
            for user, value in requests.items():
                requests_data[user].append(value)
            
            for user, value in input_tokens.items():
                input_tokens_data[user].append(value)
            
            for user, value in output_tokens.items():
                output_tokens_data[user].append(value)

plt.figure(figsize=(10, 6))
for user, values in requests_data.items():
    plt.plot(values, label=f'Requests - {user}')
plt.xlabel('Batch Number')
plt.ylabel('Requests Count')
plt.title('Requests per User')
plt.legend()
plt.tight_layout()
plt.savefig('requserperbatch.pdf', format='pdf', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
for user, values in input_tokens_data.items():
    plt.plot(values, label=f'Input Tokens - {user}')
plt.xlabel('Batch Number')
plt.ylabel('Input Tokens Count')
plt.title('Input Tokens per User')
plt.legend()
plt.tight_layout()
plt.savefig('inputtokenuserperbatch.pdf', format='pdf', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
for user, values in output_tokens_data.items():
    plt.plot(values, label=f'Output Tokens - {user}')
plt.xlabel('Batch Number')
plt.ylabel('Output Tokens Count')
plt.title('Output Tokens per User')
plt.legend()
plt.tight_layout()
plt.savefig('outputtokenuserperbatch.pdf', format='pdf', dpi=300)
plt.show()
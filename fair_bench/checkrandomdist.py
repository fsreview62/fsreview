import numpy as np

# Generate 5 random integers between 1 (inclusive) and 10 (exclusive)
random_ints = np.random.randint(1, 6, 10)
#print(random_ints)

size_array = [1,2,3,4]
left_out_len_array = np.array([258,258,258,258])
has_run_len_array = np.array([258,258,258,258])
cum_run_len_array = np.array([258,516,1032,2064])

need_max_token_num = left_out_len_array * size_array + cum_run_len_array

print(need_max_token_num)
print(need_max_token_num.max())
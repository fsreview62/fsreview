# FairServe Artifact
Experiments instructions in `fair_bench/README.md` and `fair_bench/REVISION.md`.
The prototype is build on top of S-LoRA, which is described below.

## Requirements
* CUDA 11.8 compatible GPU
  * Recommended: GPUs from the Ampere family, like the A100, which support bfloat16 operations.
  * Note: Older GPUs from the Turing family like the T4, which do not support bfloat16, are not supported.
* 1.13 <= PyTorch <= 2.0.1

## Installation
```bash
conda create -n slora python=3.9
conda activate slora 
# Optional: Install CUDA via conda for a smoother installation experience,
# but you may need to manually set the Anaconda path variables.
# conda install cuda -c nvidia/label/cuda-11.8.0
pip install torch==2.0.1
pip install -e .
```
Make sure triton==2.1.0

For more details on installing CUDA via conda, refer to the [CUDA Installation Guide by NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation).


## Instructions for Artifact Evaluation

Hardware and software requirements:
- One A10G(24GB) and one A100(80GB).
- Ubuntu installed with Pytorch 2.1.2 and Triton 2.1.0 would be recommended.

Install the S-LoRA runtime with VTC and baseline schedulers:
```
cd VTC-artifact
pip install -e .
```

# Running interaction and overload throttling
1. Launch the server with Llama-2-7b on an AWS instance of A100 GPU using FairServe scheduler. Save the output to a file (example: outinteract) to get important info and track queues.

```
conda activate slora
cd FairLLM/fair_bench
python launch_server.py --num-adapter 25 --enable-abort --scheduler fs_fair_interaction_limit_expect --rate-limit 50 > outinteract

```
You can use other scheduler options for FairServe like fs_fair, fs_fair_user_limit, fs_fair_userapp_limit, fs_fair_overload_limit etc. for benchmarking. However, at the moment every approach is combined in fs_fair_interaction_limit_expect. To disable user throttling set the --rate-limit to be very high (something like 1000). Set the app limits in fair_bench/trace.py [line 201] in function generate_requests_uniform_real. Currently it is set to 100 for all apps. Also remember to set how many requests each user is going to have using the variable max_reqs_per_user inside the divide_reqs function. This will change depending on how your synthetic trace is designed. By default it is 1000 as the benchmark uses the synthetic_trace.csv file specified inside the divide_reqs function.

In fair_bench/exp_suite.py, you can use different suites or create your own suites for benchmarking. For now we would use overload suite. The number of users is specified in adapters. If you want to work with two users, use adapters = [2]. You would also need to make sure that req_rate has request rates for all adapters or users. For example, if you are working with 50 users, then the req_rate could be req_rate = [np.random.uniform(3, 5, 50).tolist()]. 

The default number of users that will be created is 20. If you want to create more users, then you need pass another argument --num-adapter and specify how many user you want to work with. Make sure that this argument is always greater than the number of adapters that you specify within the overload suite. The duration for that suite is 60 sec and req_rate = [3,3] means 3*60 = 180 requests will be sent for two clients per minute. The duration argument is currently redundant when you are using a trace. 

To reproduce the fairness objective experiment set req_rate = [16,8]

In another terminal run experiments with workloads.

```
python run_exp.py --suite overload --output FS/all_results_overload.jsonl > output_all_res_fs_1min_rr_8_16_userlimitinf_applimit100_3
```

Once the run terminates, go to the first terminal window, and terminate the running server so that you can get the complete output in "outinteract".
You can use the FS/all_results_overload.jsonl to plot different metrics like response time, throughput, etc. Plot functions are in fair_bench/plot.

If you are timing your benchmark and don't want it to complete terminate the run in the middle. Open the output_all_res_fs_1min_rr_8_16_userlimitinf_applimit100_3 file which contains stats about the run. Get rid of all lines until you see request outputs. Those would follow this kind of output.

```
interaction_id 253 req_id 1002 request_start_time 1740005157.99 request_end_time 1740005160.28 req_time 0.4938392883012395 adapter_dir dummy-lora-7b-rank-8-1 prompt_len 50 output_len 50 sys_len 50 app 2 input99app 50 sys99app 50 output99app 50 priorityfactor 1 app_limit 100 request_latency 2.29 s, first_token_latency 0.84 s llmcalls: 1 llmcalls_made: 1
```

Then you can perform analysis. For example, if you want to find the number of responses that completed for a particular user, then check for how many requests had llmcalls == llmcalls_made. Among these, subtract the number of requests where first_token_latency -1.00 s and llmcalls == llmcalls_made. You do a quick search. Assume you are using the synthetic_trace.csv file, then if you want the number of requests that completed for user 1, completed_responses_user1 = 4 x (Responses_with_lines("llmcalls: 4 llmcalls_made: 4") - Responses_with_lines("first_token_latency -1.00 s llmcalls: 4 llmcalls_made: 4"))
completed_responses_user2 = Responses_with_lines("llmcalls: 1 llmcalls_made: 1") - Responses_with_lines("first_token_latency -1.00 s llmcalls: 1 llmcalls_made: 1")

Plotting files use functions from fair_bench/visualize.py. So in case we want to plot metrics like service received we need to make those changes over there as our definition of service is different compared to VTC.

## Acknowledgment
FairServe is build on top of [S-LoRA](https://github.com/S-LoRA/S-LoRA).


## Citations
```
@article{fairserve2025,
  title={FairServe: Ensuring Fair LLM Serving Amid Diverse Applications},
  author={Anonymous Authors},
  journal={Under Review},
  year={2025}
}
```

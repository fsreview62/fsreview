"""
To run with real mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode synthetic
default to synthetic mode.
"""
import argparse
import asyncio
import csv
import json
import numpy as np
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import List, Tuple

import pandas as pd

import aiohttp
import logging

from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import generate_requests, get_real_requests, get_trace_data_m365
sys.path.append("../bench_lora")
from slora.utils.metric import reward, attainment_func

logging.getLogger('asyncio').setLevel(logging.CRITICAL)

GB = 1024 ** 3


@dataclass
class Response:
    adapter_dir: str
    prompt_len: int
    output_len: int
    request_latency: float
    first_token_latency: float
    req_time: float
    interaction_id: int
    req_id: int 
    sys_len: int 
    app: int 
    input99app: int
    sys99app: int
    output99app: int 
    priorityfactor: int 
    app_limit: int
    llmcalls: int
    llmcalls_made: int


async def send_request(
    backend: str,
    server: str,
    req_time: float,
    req_id: str,
    model_dir: str,
    adapter_dir: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    debug: bool,
    sys_len: int,
    app: int,
    input99app: int,
    sys99app: int,
    output99app: int,
    priorityfactor: int,
    llmcalls: int,
    app_limit: int,
    llmcalls_made: int,
    interaction_id: int
) -> None:
    request_start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    headers = {"User-Agent": "Benchmark Client"}
    url = server + "/generate_stream"
    
    if backend in ["slora"]:
        data = {
            'model_dir': model_dir,
            'lora_dir': adapter_dir,
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            },
            'req_id': req_id,
            'sys_len': sys_len,
            'app': app,
            'input99app': input99app,
            'sys99app': sys99app,
            'output99app': output99app,
            'priorityfactor': priorityfactor,
            'llmcalls': llmcalls,
            'app_limit': app_limit,
            'llmcalls_made': llmcalls_made,
            'interaction_id': interaction_id
        }
    else:
        raise NotImplementedError()

    try:
        first_token_latency = None
        timeout = aiohttp.ClientTimeout(total=8 * 60)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            while True:
                async with session.post(url, headers=headers, json=data) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        if first_token_latency is None:
                            first_token_latency = time.time() - request_start_time
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                # output = json.loads(output)
                # print(output)
                
                if '\"finished\": -1' not in output:
                    break
                else:
                    first_token_latency = None
                    break
                # #     print(output)
                # #     print(json.loads(output))
                # break

        request_end_time = time.time()
        request_latency = request_end_time - request_start_time
        if first_token_latency is None: # avoid the case where the request is aborted
            first_token_latency = request_latency = -1
            print(f"Request {req_id} got throttled")
        else:
            print(f"Request {req_id} got LLM response")
    except Exception as e:
        print(f"Request {req_id} got error: {e}")
        request_end_time = time.time()
        request_latency = request_end_time - request_start_time
        if first_token_latency is None: # avoid the case where the request is aborted
            first_token_latency = request_latency = -1
    # print(f"time: {time.time()}"
    #       f"interaction_id {interaction_id} req_id {req_id} req_time {req_time} adapter_dir {adapter_dir} "
    #       f"prompt_len {prompt_len} output_len {output_len} sys_len {sys_len} app {app} input99app {input99app} sys99app {sys99app} output99app {output99app} priorityfactor {priorityfactor} app_limit {app_limit} "
    #       f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s "
    #       f"llmcalls: {llmcalls} "
    #       f"llmcalls_made: {llmcalls_made} ")
    # print(f"{req_id} ", end="")
    return Response(adapter_dir, prompt_len, output_len,
                    request_latency, first_token_latency, req_time, interaction_id, req_id, sys_len, app, input99app, sys99app, output99app, priorityfactor, app_limit, llmcalls, llmcalls_made)


async def benchmark(
    backend: str,
    server: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    debug=False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for i in range(5):
        print(f"input_requests[{i}]: {input_requests[i]}")
    for req in input_requests:
        await asyncio.sleep(start + req.req_time - time.time())
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.adapter_dir}")
        #print("req.prompt")
        task = asyncio.create_task(send_request(backend, server, req.req_time,
                                                req.req_id, req.model_dir, req.adapter_dir,
                                                req.prompt, req.prompt_len, req.output_len,
                                                debug, req.sys_len, req.app, req.input99app, req.sys99app, req.output99app, req.priorityfactor, req.llmcalls, req.app_limit, req.llmcalls_made, req.interaction_id))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    return responses


def get_adapter_dirs(num_adapters, adapter_dirs, backend=None):
    ret = []
    num_iter = num_adapters // len(adapter_dirs) + 1

    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    return ret

def get_res_stats(responses, benchmark_time, backend, request_log):
    # get throughput
    num_abort = len([x for x in responses if x.first_token_latency==-1])
    all_responses = [x for x in responses]
    responses = [x for x in responses if x.first_token_latency!= -1]
    throughput = len(responses) / benchmark_time
    users = sorted(list(set([response.adapter_dir for response in responses])))
    # print(f"users: {users}")
    df = pd.DataFrame([res.__dict__ for res in responses])
    df.to_csv(request_log)

    reqcountuser = {user: 0 for user in users}
    avgreqlatencyuser = {user: [] for user in users}
    avgtokenlatencyuser = {user: [] for user in users}
    avgoutputtokenlatencyuser = {user: [] for user in users}
    avgfirsttokenlatencyuser = {user: [] for user in users}
    throughputperuser = {user: 0 for user in users}

    for user in users:
        for res in responses:
            if res.adapter_dir == user:
                reqcountuser[res.adapter_dir]+=1
                avgreqlatencyuser[res.adapter_dir].append(res.request_latency)
                avgfirsttokenlatencyuser[res.adapter_dir].append(res.first_token_latency)
                avgtokenlatencyuser[res.adapter_dir].append(res.request_latency/(res.prompt_len + res.output_len))
                avgoutputtokenlatencyuser[res.adapter_dir].append(res.request_latency/res.output_len)

    
    for user in users:
        avgreqlatencyuser[user] = np.mean(avgreqlatencyuser[user])
        avgtokenlatencyuser[user] = np.mean(avgtokenlatencyuser[user])
        avgoutputtokenlatencyuser[user] = np.mean(avgoutputtokenlatencyuser[user])
        avgfirsttokenlatencyuser[user] = np.mean(avgfirsttokenlatencyuser[user])
    # print(responses)
    for user, reqcount in reqcountuser.items():
        throughputperuser[user] = reqcount/benchmark_time
        #print(f"user: {user}, requests processed: {reqcount}, requests/s or throughput: {throughputperuser[user]:.2f}")
    
    for user in users:
        print(f"user: {user}, requests: {reqcountuser[user]}, throughput: {throughputperuser[user]} req/s, average request latency: {avgreqlatencyuser[user]}, average first token latency: {avgfirsttokenlatencyuser[user]} average token latency: {avgtokenlatencyuser[user]}, avg output token latency: {avgoutputtokenlatencyuser[user]} ")
        #print(f"user: {user}, average token latency: {avgtokenlatencyuser[user]}")
    
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Aborted Request: {num_abort}")
    print(f"Throughput: {throughput:.2f} requests/s")

    # compute the latency statistics.
    avg_latency = np.mean([x.request_latency for x in responses])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        x.request_latency / (x.prompt_len + x.output_len)
        for x in responses
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean([
        x.request_latency / x.output_len
        for x in responses
    ])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")

    # compute the first token latency
    #first_token_latency = [x.request_latency for x in responses]
    first_token_latency = [x.first_token_latency for x in responses]
    avg_first_token_latency = np.mean(first_token_latency)
    print(f"Average first token latency: {avg_first_token_latency:.2f} s")
    print(f"90 percentile first token latency: < {np.percentile(first_token_latency, 90):.2f} s")
    print(f"50 percentile first token latency: < {np.percentile(first_token_latency, 50):.2f} s")

    # compute the attainment
    attainment = [attainment_func(x.request_latency) for x in responses]
    avg_attainment = np.mean(attainment)
    print(f"Average attainment: {avg_attainment:.2f}")

    # compute the per adapter first token latency (ftl) and attainment
    ftl_per_adapter = {}
    attainment_per_adapter = {}
    for x in responses:
        if x.adapter_dir not in ftl_per_adapter:
            #ftl_per_adapter[x.adapter_dir] = [x.request_latency]
            ftl_per_adapter[x.adapter_dir] = [x.first_token_latency]
            attainment_per_adapter[x.adapter_dir] = [attainment_func(x.request_latency)]
        else:
            #ftl_per_adapter[x.adapter_dir].append(x.request_latency)
            ftl_per_adapter[x.adapter_dir].append(x.first_token_latency)
            attainment_per_adapter[x.adapter_dir].append(attainment_func(x.request_latency))
    for k, v in ftl_per_adapter.items():
        print(f"Average first token latency ({len(v)}) for adapter {k}: {np.mean(v)} s")

    # dump results
    result = {"total_time": benchmark_time, "num_abort": num_abort,
              "throughput": throughput,
              "avg_latency": avg_latency, "avg_per_token_latency": avg_per_token_latency,
              "avg_per_output_token_latency": avg_per_output_token_latency,
              "avg_first_token_latency": avg_first_token_latency,
              "avg_attainment": avg_attainment}#,
    
    # Adding user-specific data to the result dictionary
    for user in users:
        result[f"{user}_requests"] = reqcountuser.get(user, 0)
        result[f"{user}_throughput"] = throughputperuser.get(user, 0)
        result[f"{user}_avg_request_latency"] = avgreqlatencyuser.get(user, 0)
        result[f"{user}_avg_firsttoken_latency"] = avgfirsttokenlatencyuser.get(user, 0)
        result[f"{user}_avg_token_latency"] = avgtokenlatencyuser.get(user, 0)
        result[f"{user}_avg_output_token_latency"] = avgoutputtokenlatencyuser.get(user, 0)
    
    result["responses"] = [asdict(x) for x in all_responses]

    res = {"config": to_dict(config), "result": result}
    
    return res


def run_exp(model_setting, backend, server, config, output, seed=42, debug=False, input="", app_limit=None, request_log=None, app_stats="avg"):
    print([(k, v) for k, v in zip(BenchmarkConfig._fields, config)])

    num_adapters, alpha, req_rate, cv, duration, input_range, output_range, on_off, mode = config
    # assert duration >= 30
    base_model = BASE_MODEL[model_setting]
    adapter_dirs = LORA_DIR[model_setting]
    if mode == "real":
        print("*** num_adapters, cv and alpha are not used in real mode ***")
        pkl_file_path = "real_trace.pkl"
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, "rb") as pkl_file:
                obj = pickle.load(pkl_file)
            adapter_dirs, requests = obj[0], obj[1]

            data_str = repr(obj)
            with open('real_trace.txt', 'w') as file:
                file.write(data_str)
        else:
            adapter_dirs, requests = get_real_requests(trace_file="dummy_chat_conv_20231016.json",
                                                       req_rate=req_rate, duration=duration,
                                                       base_model=base_model, adapter_dirs=adapter_dirs,
                                                       input_range=input_range, output_range=output_range,
                                                       seed=seed)
            with open(pkl_file_path, "wb") as pkl_file:
                pickle.dump([adapter_dirs, requests], pkl_file)
    elif mode == "real_m365_trace":
        requests, num_adapters = get_trace_data_m365(base_model, adapter_dirs, input, app_limit, app_stats)
    else:
        # print(requests)
        adapter_dirs = get_adapter_dirs(num_adapters, adapter_dirs)
        adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(num_adapters)]
        print(f"adapter_dirs: {adapter_dirs}")
        if num_adapters == 0:
            adapter_dirs = [(base_model, None)]
            num_adapters = 1
        
        num_apps = 2
        requests = generate_requests(num_adapters, alpha, req_rate, cv, duration,
                                    input_range, output_range, on_off, mode, adapter_dirs, num_apps,
                                    seed=seed)
        print(f"requests len: {len(requests)}")
        # for i in range(5):
        #     print(f"requests[{i}]: {requests[i]}")
    print(f"requests len: {len(requests)}")
    for i in range(20):
        print(f"requests[{i}]: {requests[i]}")
    avg_prompt_len = np.mean([req.prompt_len for req in requests])
    avg_output_len = np.mean([req.output_len for req in requests])
    avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
    max_len = np.max([req.prompt_len + req.output_len for req in requests])
    print("num_adapters", len(adapter_dirs), "num_requests", len(requests), "avg_len:", avg_len, "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len, "max_len:", max_len)
       
    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    # benchmark
    benchmark_start_time = time.time()
    print(f"type(requests) :{type(requests)}")
    print(f"type(requests[0]): {type(requests[0])}")
    #print(requests)
    # print(f"type(requests[0][0]): {type(requests[0][0])}")
    # print(f"type(requests[0][1]): {type(requests[0][1])}")
    # print(f"type(requests[0][2]): {type(requests[0][2])}")
    # print(f"type(requests[0][3]): {type(requests[0][3])}")
    # print(f"type(requests[0][4]): {type(requests[0][4])}")

    responses = asyncio.run(benchmark(backend, server, requests, debug))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    res = get_res_stats(responses, benchmark_time, backend, request_log)

    os.makedirs("/".join(output.split("/")[:-1]), exist_ok=True)
    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")
        print(f"Written to file {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="slora")
    parser.add_argument("--suite", type=str, default="default")

    parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--server", type=str, default="http://localhost:8000")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--input", type=str, default="")

    parser.add_argument("--app-limit", type=int, default=200)

    parser.add_argument("--request-log", type=str, required=True)

    parser.add_argument("--app-stat", type=str, default="avg", choices=["99", "avg", "min", "max"])

    args = parser.parse_args()

    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.suite}.jsonl"
    if args.debug:
        args.output = "debug_" + args.output

    suites = get_all_suites(debug=args.debug, suite=args.suite)

    print(f"len(suites): {len(suites)}")
    print(f"the suites: {suites}")

    if not args.append:
        os.system(f"rm {args.output}")
        results = []
    else:
        with open(args.output, "r") as f:
            lines = f.readlines()
        results = [json.loads(line)["config"] for line in lines]

    for config in tqdm(suites, desc="suites"):
        if to_dict(config) not in results:
            stats = run_exp(args.model_setting, args.backend, args.server, config,
                            args.output, args.seed, args.debug, args.input, args.app_limit,
                            args.request_log, args.app_stat)

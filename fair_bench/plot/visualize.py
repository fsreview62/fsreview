import argparse
import json
import numpy as np

from plot_utils import plot
from collections import defaultdict


FONTSIZE = 25
MARKERSIZE = 5
legend_x = 0.44
legend_y = 1.15
ylabel_x = -0.15
ylabel_y = 0.5

MARKERS = ["s", "o", "v"]

adapter_dirs = ['dummy-lora-7b-rank-8-0', 'dummy-lora-7b-rank-8-1', 'dummy-lora-7b-rank-8-2', 'dummy-lora-7b-rank-8-3', 'dummy-lora-7b-rank-8-4', 'dummy-lora-7b-rank-8-5', 'dummy-lora-7b-rank-8-6', 'dummy-lora-7b-rank-8-7', 'dummy-lora-7b-rank-8-8', 'dummy-lora-7b-rank-8-9', 'dummy-lora-7b-rank-8-10']

systokens = {}
input99percuser = {}
sys99percuser = {}
output99percuser = {}
priorityfactoruser = {}
systoken_len = [10,200,88,150,120,160,174,196,144,130,105]
llmagent = [1,10,4,3,1,5,6,7,9,10,5]
tokenperagent = [10,20,22,50,120,32,29,28,16,13,21]
#input99perc = [496,116,512,512,512,512,512,512,512,512,512]
input99perc = [50,580,512,512,512,512,512,512,512,512,512]
#output99perc = [512,512,512,512,512,512,512,512,512,512,512]
output99perc = [258,258,512,512,512,512,512,512,512,512,512]
sys99perc = [100,2000,880,1500,1200,1600,1740,1960,1440,1300,1050]
priorityfactor = [1,1,2,5,1,1,1,1,1,1,1]

for i in range(len(adapter_dirs)):
    #print(i)
    systokens[adapter_dirs[i]] = int(llmagent[i%10] * tokenperagent[i%10])
    input99percuser[adapter_dirs[i]] = int(input99perc[i%10])
    sys99percuser[adapter_dirs[i]] = int(sys99perc[i%10])
    output99percuser[adapter_dirs[i]] = int(output99perc[i%10])
    priorityfactoruser[adapter_dirs[i]] = int(priorityfactor[i%10])


def to_client_name(adapter_name):
    if adapter_name == "dummy-lora-7b-rank-8-0":
        return "Client 1"
    if adapter_name == "dummy-lora-7b-rank-8-1":
        return "Client 2"
    if adapter_name == "dummy-lora-7b-rank-8-2":
        return "Client 3"
    if adapter_name == "dummy-lora-7b-rank-8-3":
        return "Client 4"
    return adapter_name


def get_req_rate_over_time(responses, T, window, x_ticks, users, warmup=0):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2 + warmup
            r = x + window / 2 + warmup
            for response in responses:
                if response["adapter_dir"] == user_name:
                    req_time = response["req_time"]
                    num_token = response["output_len"]
                    if l <= req_time and req_time <= r:
                        # y[-1][i] += num_token
                        y[-1][i] += response["prompt_len"] + 2 * response["output_len"]
            y[-1][i] /= window
    return y


def get_throughput_over_time(responses, T, window, x_ticks, users):
    y = []
    sum = 0
    for r in responses:
        if r["first_token_latency"] != -1:
            sum += 1
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    num_token = response["output_len"]
                    overlap = max(min(r, end_time) - max(l, start_time), 0)
                    if end_time == start_time: # for aborted requests
                        assert response["first_token_latency"] == -1, "first_token_latency should be -1 for aborted requests"
                        y[-1][i] += 0
                    else:
                        y[-1][i] += num_token * overlap / (end_time - start_time)
            y[-1][i] /= window
    return y


def get_overall_throughput(result):
    responses = result["responses"]
    num_token = sum([response["prompt_len"] + response["output_len"] for response in responses])
    return num_token / result["total_time"]


def cost_func(input_len, output_len, func_type="linear"):
    if func_type == "linear":
        return input_len + 2*output_len
    elif func_type == "profile":
        x, y = input_len, output_len
        return (0.21001891 * x + 0.10075395 * y +
                0.0039868 * x * y + 0.00324733 * y * y + 1.14601454)
    else:
        raise Exception(f"unrecognized cost function {func_type}")

def cost_func_fair(input_len, input_99_perc, system_len, system_99_perc, output_len, output_99_perc, func_type="linear"):
    if func_type == "linear":
        return (input_len/input_99_perc) + (system_len/system_99_perc) + 2*(output_len/output_99_perc)

def get_wasted_tokens(responses):
    y = []
    user_wasted_tokens = defaultdict(int)
    user_interaction_aborted = defaultdict(set)
    user_requests_aborted = defaultdict(list)
    interaction_to_reqs = defaultdict(list)
    for response in responses:
        if response["first_token_latency"] == -1:
            user_interaction_aborted[response["adapter_dir"]].add(response["interaction_id"])
            user_requests_aborted[response["adapter_dir"]].append(response["req_id"])
        interaction_to_reqs[response["interaction_id"]].append(response)
    
    for user in user_interaction_aborted.keys():
        for aborted_interaction in user_interaction_aborted[user]:
            for req in interaction_to_reqs[aborted_interaction]:
                #print(f"user: {user} aborted_interaction: {aborted_interaction}, req: {req} ")
                if req["first_token_latency"]!=-1:
                    #print(f"user: {req['adapter_dir']}, wasted token interaction: {req['interaction_id']}, wasted token request: {req['req_id']} ")
                    user_wasted_tokens[req["adapter_dir"]]+= req['prompt_len']

    return user_wasted_tokens, user_interaction_aborted, user_requests_aborted

def get_service_over_time_fair(responses, T, window, x_ticks, users, func_type="linear"):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    if end_time == start_time: # for aborted requests
                        assert response["first_token_latency"] == -1, "first_token_latency should be -1 for aborted requests"
                        y[-1][i] += 0
                    else:
                        service = cost_func_fair(response['prompt_len'] - response['sys_len'], response['input99app'], response['sys_len'], response['sys99app'], response["output_len"], response['output99app'], func_type=func_type)
                        overlap = max(min(r, end_time) - max(l, start_time), 0)
                        y[-1][i] += service * overlap / (end_time - start_time)
            y[-1][i] /= window
    return y

def get_full_service_over_time_fair(responses, T, window, x_ticks, func_type="linear"):
    y = []
    #for i, user_name in enumerate(users):
    y.append([0] * len(x_ticks))
    for i, x in enumerate(x_ticks):
        l = x - window / 2
        r = x + window / 2
        for response in responses:
            #if response["adapter_dir"] == user_name:
            start_time = response["req_time"] + response["first_token_latency"]
            end_time = response["req_time"] + response["request_latency"]
            if end_time == start_time: # for aborted requests
                assert response["first_token_latency"] == -1, "first_token_latency should be -1 for aborted requests"
                y[-1][i] += 0
            else:
                service = cost_func_fair(response['prompt_len'] - response['sys_len'], response['input99app'], response['sys_len'], response['sys99app'], response["output_len"], response['output99app'], func_type=func_type)
                overlap = max(min(r, end_time) - max(l, start_time), 0)
                y[-1][i] += service * overlap / (end_time - start_time)
        y[-1][i] /= window
    return y

def get_full_throughput_over_time(responses, T, window, x_ticks):
    y = []
    sum = 0
    for r in responses:
        if r["first_token_latency"] != -1:
            sum += 1
    #for i, user_name in enumerate(users):
    y.append([0] * len(x_ticks))
    for i, x in enumerate(x_ticks):
        l = x - window / 2
        r = x + window / 2
        for response in responses:
            #if response["adapter_dir"] == user_name:
            start_time = response["req_time"] + response["first_token_latency"]
            end_time = response["req_time"] + response["request_latency"]
            num_token = response["output_len"]
            overlap = max(min(r, end_time) - max(l, start_time), 0)
            if end_time == start_time: # for aborted requests
                assert response["first_token_latency"] == -1, "first_token_latency should be -1 for aborted requests"
                y[-1][i] += 0
            else:
                y[-1][i] += num_token * overlap / (end_time - start_time)
        y[-1][i] /= window
    return y

def get_service_over_time(responses, T, window, x_ticks, users, func_type="linear"):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    if end_time == start_time: # for aborted requests
                        assert response["first_token_latency"] == -1, "first_token_latency should be -1 for aborted requests"
                        y[-1][i] += 0
                    else:
                        service = cost_func(response['prompt_len'],response["output_len"], func_type=func_type)
                        overlap = max(min(r, end_time) - max(l, start_time), 0)
                        y[-1][i] += service * overlap / (end_time - start_time)
            y[-1][i] /= window
    return y

def get_response_time_over_time(responses, T, window, x_ticks, users):
    y = []
    for j, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            cnt = 0
            for response in responses:
                if response["adapter_dir"] == user_name:
                    req_time = response["req_time"] + response["first_token_latency"]
                    response_time = response["first_token_latency"]
                    # if j == 2 and req_time > 454 and req_time < 515:
                    #     print(response)
                    if l <= req_time and req_time <= r:
                        if not response_time == -1: # for aborted requests
                            y[-1][i] += response_time
                            cnt += 1
            if cnt == 0:
                y[-1][i] = None
            else:
                #print(f"user: {j}, i: {i}, cnt: {cnt}, response time: {y[-1][i]}")
                y[-1][i] /= cnt
                print(f"user: {j}, i: {i}, cnt: {cnt}, response time: {y[-1][i]}")
    return y

def get_full_response_time_over_time(responses, T, window, x_ticks):
    y = []
    #for j, user_name in enumerate(users):
    y.append([0] * len(x_ticks))
    for i, x in enumerate(x_ticks):
        l = x - window / 2
        r = x + window / 2
        cnt = 0
        for response in responses:
            #if response["adapter_dir"] == user_name:
            req_time = response["req_time"] + response["first_token_latency"]
            response_time = response["first_token_latency"]
            # if j == 2 and req_time > 454 and req_time < 515:
            #     print(response)
            if l <= req_time and req_time <= r:
                if not response_time == -1: # for aborted requests
                    y[-1][i] += response_time
                    cnt += 1
        if cnt == 0:
            y[-1][i] = None
        else:
            #print(f"user: {j}, i: {i}, cnt: {cnt}, response time: {y[-1][i]}")
            y[-1][i] /= cnt
            #print(f"user: {j}, i: {i}, cnt: {cnt}, response time: {y[-1][i]}")
    return y

def get_acc_service(responses, T, window, x_ticks, users, warmup=0):
    y = []
    # print(f"there are {len(users)} users")
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = 0 + warmup
            r = x + warmup
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    if end_time < 1 or end_time - start_time < 1e-5:
                        continue
                    service = cost_func(response['prompt_len'],response["output_len"])
                    overlap = max(min(r, end_time) - max(l, start_time), 0)
                    y[-1][i] += service * overlap / (end_time - start_time)
    # y[i][j]: accumulated service of client i at time tick j
    return y


def get_acc_service_diff_with_first_client(responses, T, window, x_ticks, users, warmup=0):
    diff = get_acc_service(responses, T, window, x_ticks, users, warmup=warmup)
    for i in range(1, len(users)):
        diff[i] = list(np.array(diff[i]) - np.array(diff[0]))
    diff[0] = list(np.array(diff[0]) * 0)
    # diff[i][j]: the delta of accumulated service of client i compare to client 0 at time tick j
    return diff


def get_acc_service_diff_over_time(responses, T, window, x_ticks, users, warmup=0):
    y = get_acc_service(responses, T, window, x_ticks, users, warmup=warmup)
    # compute max difference in service over time
    max_diff = []
    for i in range(len(x_ticks)):
        max_diff.append(max([y[j][i] for j in range(len(users))]) - min([y[j][i] for j in range(len(users))]))
    return max_diff


def get_service_diff_over_time(responses, T, window, x_ticks, users, req_rate, warmup=0, func_type="linear"):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2 + warmup
            r = x + window / 2 + warmup
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    if end_time < 1 or end_time - start_time < 1e-5:
                        continue
                    service = cost_func(response['prompt_len'], response["output_len"], func_type=func_type)
                    overlap = max(min(r, end_time) - max(l, start_time), 0)
                    y[-1][i] += service * overlap / (end_time - start_time)
            y[-1][i] /= window
    # compute max difference in service over time
    # max_diffs = []
    # for i in range(len(x_ticks)):
    #     max_service = max([y[j][i] for j in range(len(users))])
    #     max_diff = float("-inf")
    #     for j in range(len(users)):
    #         if req_rate[j][i] < y[j][i]:
    #             # tricky cases: backlogged or just noise
    #             max_diff = max(max_diff, min(max_service - y[j][i], y[j][i] - req_rate[j][i]))
    #         else:
    #             max_diff = max(max_diff, min(max_service - y[j][i], req_rate[j][i] - y[j][i]))
    #     max_diffs.append(max_diff)
    # return max_diffs

    # max_diffs = []
    # for i in range(len(x_ticks)):
    #     max_service = max([y[j][i] for j in range(len(users))])
    #     max_diff = float("-inf")
    #     for j in range(len(users)):
    #         if req_rate[j][i] > max_diff:
    #             max_diff = max(max_diff, min(max_service - y[j][i], req_rate[j][i] - y[j][i]))
    #     max_diffs.append(max_diff)
    # return max_diffs

    sum_diffs = []
    for i in range(len(x_ticks)):
        max_service = max([y[j][i] for j in range(len(users))])
        sum_diff = 0
        for j in range(len(users)):
            # print("user", j)
            # print(max_service - y[j][i], req_rate[j][i] - y[j][i], req_rate[j][i])
            if req_rate[j][i] < y[j][i]:
                # tricky cases: backlogged or just noise
                sum_diff += min(max_service - y[j][i], y[j][i] - req_rate[j][i])
            else:
                sum_diff += min(max_service - y[j][i], req_rate[j][i] - y[j][i])
        # print(f"x at {i}", sum_diff)
        sum_diffs.append(sum_diff)
    return sum_diffs


def gen_quant_fairness_table(schedulers, service_diffs, throughputs, figname):
    tab = (
        "\\begin{tabular}{c|cccc}\n"
        "\\toprule\n"
        "Scheduler & Max Diff & Avg Diff & Diff Var & Throughput \\\\ \n"
        "\\midrule\n"
    )
    for i, scheduler in enumerate(schedulers):
        max_diff = max(service_diffs[i])
        avg_diff = np.mean(service_diffs[i])
        diff_var = np.var(service_diffs[i])
        tab += f"{scheduler} & {max_diff:.2f} & {avg_diff:.2f} & {diff_var:.2f} & {throughputs[i]:.0f}\\\\ \n"
    tab += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    with open(f"{figname}.tex", "w") as f:
        f.write(tab)
    print(f"Write tex to {figname}.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    exps = []
    with open(args.input, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps.append({})
            exps[-1]["config"] = json.loads(line)["config"]
            exps[-1]["result"] = json.loads(line)["result"]

    # get data points
    for exp in exps:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = list(set([response["adapter_dir"] for response in responses]))

        req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users)
        service = get_service_over_time(responses, T, window, x_ticks, users)
        throughput = get_throughput_over_time(responses, T, window, x_ticks, users)
        response_time = get_response_time_over_time(responses, T, window, x_ticks, users)


    # plot
    plot(users, x_ticks, req_rate, "time progression (s)", "req_rate (token/s)", "req_rate")
    plot(users, x_ticks, throughput, "time progression (s)", "throughput (token/s)", "throughput")
    plot(users, x_ticks, service, "time progression (s)", "service (token/s)", "service")
    plot(users, x_ticks, response_time, "time progression (s)", "response_time (s)", "response_time")

    cnt = {}
    for user_name in users:
        cnt[user_name] = 0

    for response in responses:
        cnt[response["adapter_dir"]] += 1
    print(cnt)



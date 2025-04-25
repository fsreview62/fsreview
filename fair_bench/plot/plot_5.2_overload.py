import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# from plot.plot_utils import plot
from visualize import (get_req_rate_over_time, get_throughput_over_time, get_service_over_time,get_service_over_time_fair,
                       get_response_time_over_time,get_full_service_over_time_fair, get_full_throughput_over_time, get_full_response_time_over_time, get_wasted_tokens, to_client_name,
                       FONTSIZE, MARKERSIZE, legend_x, legend_y, ylabel_x, ylabel_y)


def plot(names, x, ys, x_label, y_label, figname):
    legends = []
    curves = []
    fig, ax = plt.subplots()
    
    for i, (name, y) in enumerate(zip(names, ys)):
        curves.append(ax.plot(x, y, color=f"C{i}", marker=".", markersize=MARKERSIZE)[0])
        legends.append(to_client_name(name))

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")
    if figname == "sec5.2_overload_service":
        y_ticks = np.arange(0,14,2)
        ax.set_yticks(y_ticks)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=FONTSIZE)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

def plot_full(x, ys, x_label, y_label, figname):
    legends = []
    curves = []
    names = ["VTC"]
    fig, ax = plt.subplots()
    for i, (name, y) in enumerate(zip(names, ys)):
        curves.append(ax.plot(x, y, color=f"C{i}", marker=".", markersize=MARKERSIZE)[0])
        legends.append(to_client_name(name))

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")
    if figname == "Eval_overload_full_service":
        y_ticks = np.arange(0,4,1)
        ax.set_yticks(y_ticks)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=FONTSIZE)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

def plot_methods(names ,x, ys, x_label, y_label, figname):
    legends = []
    curves = []
    #names = ["VTC"]
    fig, ax = plt.subplots()
    ys = list(ys.values())
    #print(ys)
    for i, (name, y) in enumerate(zip(names, ys)):
        curves.append(ax.plot(x, y, color=f"C{i}", marker=".", markersize=MARKERSIZE)[0])
        legends.append(to_client_name(name))

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")
    if figname == "Eval_overload_full_service":
        y_ticks = np.arange(0,4,1)
        ax.set_yticks(y_ticks)

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=FONTSIZE)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #downsampled_data_10000_120_seconds.csv_fs_fair_interaction_limit_1000.jsonl
    #downsampled_data_10000_120_seconds.csv_vtc_fair.jsonl
    parser.add_argument("--input_vtc", type=str, default="../../outputs/downsampled_data_10000_120_seconds.csv_vtc_fair.jsonl")
    parser.add_argument("--input_fs", type=str, default="../../outputs/downsampled_data_10000_120_seconds.csv_fs_fair_interaction_limit_100.jsonl")
    # parser.add_argument("--input_rpm", type=str, default="../all_results_overload.jsonl")
    args = parser.parse_args()

    exps = []
    exps_fs = []
    with open(args.input_vtc, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps.append({})
            exps[-1]["config"] = json.loads(line)["config"]
            exps[-1]["result"] = json.loads(line)["result"]

    with open(args.input_fs, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps_fs.append({})
            exps_fs[-1]["config"] = json.loads(line)["config"]
            exps_fs[-1]["result"] = json.loads(line)["result"]

    # get data points

    total_service_list = {}
    total_throughput_list = {}
    total_response_time_list = {}
    user_wasted_tokens_list = {}
    user_interaction_aborted_list = {}
    user_requests_aborted_list = {}

    for exp in exps:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = sorted(list(set([response["adapter_dir"] for response in responses])))

        #req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users)
        #throughput = get_throughput_over_time(responses, T, window, x_ticks, users)
        #service = get_service_over_time(responses, T, window, x_ticks, users)
        #service = get_service_over_time_fair(responses, T, window, x_ticks, users)
        total_service = get_full_service_over_time_fair(responses, T, window, x_ticks)
        total_throughput = get_full_throughput_over_time(responses, T, window, x_ticks)
        total_response_time = get_full_response_time_over_time(responses, T, window, x_ticks)
        #response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
        user_wasted_tokens, user_interaction_aborted, user_requests_aborted = get_wasted_tokens(responses)

        print("---------------Wasted tokens stats-----------")
        for key, val in user_wasted_tokens.items():
            print(f"user: {key}, wasted_tokens: {val}")
        
        print("---------------aborted interaction stats-----------")
        for key, val in user_interaction_aborted.items():
            print(f"user: {key}, len: {len(val)}, interaction_ids: {val}")

        print("---------------aborted requests stats-----------")
        for key, val in user_requests_aborted.items():
            print(f"user: {key},  len: {len(val)}, req_ids: {val}")
       
        total_service_list["VTC"] = total_service[0]
        total_throughput_list["VTC"] = total_throughput[0]
        total_response_time_list["VTC"] = total_response_time[0]
        user_wasted_tokens_list["VTC"] = user_wasted_tokens
        user_interaction_aborted_list["VTC"] = user_interaction_aborted
        user_requests_aborted_list["VTC"] = user_requests_aborted



    for exp in exps_fs:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = sorted(list(set([response["adapter_dir"] for response in responses])))

        #req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users)
        #throughput = get_throughput_over_time(responses, T, window, x_ticks, users)
        #service = get_service_over_time(responses, T, window, x_ticks, users)
        #service = get_service_over_time_fair(responses, T, window, x_ticks, users)
        total_service = get_full_service_over_time_fair(responses, T, window, x_ticks)
        total_throughput = get_full_throughput_over_time(responses, T, window, x_ticks)
        total_response_time = get_full_response_time_over_time(responses, T, window, x_ticks)
        #response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
        user_wasted_tokens, user_interaction_aborted, user_requests_aborted = get_wasted_tokens(responses)

        print("---------------FS Wasted tokens stats-----------")
        for key, val in user_wasted_tokens.items():
            print(f"user: {key}, wasted_tokens: {val}")
        
        print("---------------FS aborted interaction stats-----------")
        for key, val in user_interaction_aborted.items():
            print(f"user: {key}, len: {len(val)}, interaction_ids: {val}")

        print("---------------FS aborted requests stats-----------")
        for key, val in user_requests_aborted.items():
            print(f"user: {key},  len: {len(val)}, req_ids: {val}")

        total_service_list["FS"] = total_service[0]
        total_throughput_list["FS"] = total_throughput[0]
        total_response_time_list["FS"] = total_response_time[0]
        user_wasted_tokens_list["FS"] = user_wasted_tokens
        user_interaction_aborted_list["FS"] = user_interaction_aborted
        user_requests_aborted_list["FS"] = user_requests_aborted
    # plot
    #plot(users, x_ticks, req_rate, "Time (s)", "Request Rate (token/s)", "sec5.2_overload_req_rate")
    #plot(users, x_ticks, throughput, "Time (s)", "Throughput (token/s)", "sec5.2_overload_throughput")
    #plot(users, x_ticks, service, "Time (s)", "Service (Token ratio/s)", "sec5.2_overload_service")
    #plot_full(x_ticks, total_service, "Time (s)", "Service (Token ratio/s)", "Eval_overload_full_service")
    #plot(users, x_ticks, response_time, "Time (s)", "Response Time (s)", "sec5.2_overload_response_time")]
    names = ["VTC", "FS"]
    plot_methods(names, x_ticks, total_service_list, "Time (s)", "Service (Token ratio/s)", "Eval_overload_full_service")
    plot_methods(names, x_ticks, total_throughput_list, "Time (s)", "Throughput (Token/s)", "Eval_overload_full_throughput")
    plot_methods(names, x_ticks, total_response_time_list, "Time (s)", "Response Time (s)", "Eval_overload_full_response")
    # cnt = {}
    # for user_name in users:
    #     cnt[user_name] = 0

    # for response in responses:
    #     cnt[response["adapter_dir"]] += 1
    #print(cnt)

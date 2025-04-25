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
    #downsampled_data_10000_120_seconds.csv___lshare_fair___50___2000.jsonl
    #downsampled_data_10000_rows_first_arrivals.csv___lshare_fair___10___2000.jsonl
    #downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit___10___2000.jsonl
    #downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit_worpmdebt___50___2000.jsonl
    #downsampled_data_10000_rows_first_arrivals.csv___fs_fair_odt_wsc_limit___50___2000.jsonl
    #downsampled_data_10000_rows_first_arrivals.csv_lshare_fair_100.jsonl
    parser.add_argument("--input1", type=str, default="../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___lshare_fair___50___2000.jsonl")
    parser.add_argument("--input2", type=str, default="../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_wsc___50___2000.jsonl")
    parser.add_argument("--input3", type=str, default="../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_odt_wsc_limit___50___2000.jsonl")
    parser.add_argument("--input4", type=str, default="../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_interaction_limit___50___2000.jsonl")
    parser.add_argument("--input5", type=str, default="../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_wsc_limit___100___2000.jsonl")
    #parser.add_argument("--input5", type=str, default="../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___fs_fair_wsc_limit___100___2000.jsonl")
    #parser.add_argument("--input6", type=str, default="../../outputs/new/downsampled_data_10000_rows_first_arrivals.csv___vtc_fair___50___2000.jsonl")
    args = parser.parse_args()

    names = ["RPM", "FS (W)", "FS (W+O)", "FS (W+O+I)", "FS (W+L)", "VTC"]

    exps_1 = []
    exps_2 = []
    exps_3 = []
    exps_4 = []
    exps_5 = []
    exps_6 = []
    with open(args.input1, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps_1.append({})
            exps_1[-1]["config"] = json.loads(line)["config"]
            exps_1[-1]["result"] = json.loads(line)["result"]

    with open(args.input2, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps_2.append({})
            exps_2[-1]["config"] = json.loads(line)["config"]
            exps_2[-1]["result"] = json.loads(line)["result"]

    with open(args.input3, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps_3.append({})
            exps_3[-1]["config"] = json.loads(line)["config"]
            exps_3[-1]["result"] = json.loads(line)["result"]

    with open(args.input4, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps_4.append({})
            exps_4[-1]["config"] = json.loads(line)["config"]
            exps_4[-1]["result"] = json.loads(line)["result"]
    
    with open(args.input5, "r") as f:
        lines = f.readlines()
        for line in lines:
            exps_5.append({})
            exps_5[-1]["config"] = json.loads(line)["config"]
            exps_5[-1]["result"] = json.loads(line)["result"]
    
    # with open(args.input6, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         exps_6.append({})
    #         exps_6[-1]["config"] = json.loads(line)["config"]
    #         exps_6[-1]["result"] = json.loads(line)["result"]
    # get data points

    total_service_list = {}
    total_throughput_list = {}
    total_response_time_list = {}
    user_wasted_tokens_list = {}
    user_interaction_aborted_list = {}
    user_requests_aborted_list = {}

    for exp in exps_1:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        print(type(responses[0]))
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = sorted(list(set([response["adapter_dir"] for response in responses])))

        total_service = get_full_service_over_time_fair(responses, T, window, x_ticks)
        total_throughput = get_full_throughput_over_time(responses, T, window, x_ticks)
        total_response_time = get_full_response_time_over_time(responses, T, window, x_ticks)
        #response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
        user_wasted_tokens, user_interaction_aborted, user_requests_aborted = get_wasted_tokens(responses)

        print(f"---------------{names[0]} Wasted tokens stats-----------")
        #total_wasted_tokens = 0
        # for key, val in user_wasted_tokens.items():
        #     print(f"user: {key}, wasted_tokens: {val}")
        mean_wasted_tokens = 0
        percentile_5_wasted_tokens = 0
        percentile_95_wasted_tokens = 0
        if len(user_wasted_tokens) > 0:
            wasted_tokens_array = np.array(list(user_wasted_tokens.values()))
            mean_wasted_tokens = np.mean(wasted_tokens_array)
            percentile_5_wasted_tokens = np.percentile(wasted_tokens_array, 5)
            percentile_95_wasted_tokens = np.percentile(wasted_tokens_array, 95)

            print(f"Mean of wasted tokens: {mean_wasted_tokens}")
            print(f"5th percentile of wasted tokens: {percentile_5_wasted_tokens}")
            print(f"95th percentile of wasted tokens: {percentile_95_wasted_tokens}")     
        input1_wasted_tokens_stats = [mean_wasted_tokens, percentile_5_wasted_tokens, percentile_95_wasted_tokens]
        
        print(f"---------------{names[0]} aborted interaction stats-----------")
        mean_user_interaction_aborted = 0
        percentile_5_user_interaction_aborted = 0
        percentile_95_user_interaction_aborted = 0
        if len(user_interaction_aborted) > 0:
            user_int_aborted_vals = []
            for key, val in user_interaction_aborted.items():
                #print(f"user: {key}, len: {len(val)}")#, interaction_ids: {val}")
                user_int_aborted_vals.append(len(val))
      
            user_interaction_aborted_array = np.array(user_int_aborted_vals)
            mean_user_interaction_aborted = np.mean(user_interaction_aborted_array)
            percentile_5_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 5)
            percentile_95_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 95)

            print(f"Mean of user_interaction_aborted: {mean_user_interaction_aborted}")
            print(f"5th percentile of user_interaction_aborted: {percentile_5_user_interaction_aborted}")
            print(f"95th percentile of user_interaction_aborted: {percentile_95_user_interaction_aborted}")
        input1_aborted_interaction_stats = [mean_user_interaction_aborted, percentile_5_user_interaction_aborted, percentile_95_user_interaction_aborted]
        
        print(f"---------------{names[0]} aborted requests stats-----------")
        mean_user_requests_aborted = 0
        percentile_5_user_requests_aborted = 0
        percentile_95_user_requests_aborted = 0
        if len(user_requests_aborted) > 0:
            user_req_aborted_vals = []
            for key, val in user_requests_aborted.items():
                #print(f"user: {key},  len: {len(val)}")#, req_ids: {val}")
                user_req_aborted_vals.append(len(val))

            user_requests_aborted_array = np.array(user_req_aborted_vals)
            mean_user_requests_aborted = np.mean(user_requests_aborted_array)
            percentile_5_user_requests_aborted = np.percentile(user_requests_aborted_array, 5)
            percentile_95_user_requests_aborted = np.percentile(user_requests_aborted_array, 95)

            print(f"Mean of user_requests_aborted: {mean_user_requests_aborted}")
            print(f"5th percentile of user_requests_aborted: {percentile_5_user_requests_aborted}")
            print(f"95th percentile of user_requests_aborted: {percentile_95_user_requests_aborted}")

        input1_aborted_req_stats = [mean_user_requests_aborted, percentile_5_user_requests_aborted, percentile_95_user_requests_aborted]
        # user_wasted_tokens_list[names[0]] = user_wasted_tokens
        # user_interaction_aborted_list[names[0]] = user_interaction_aborted
        # user_requests_aborted_list[names[0]] = user_requests_aborted

    for exp in exps_2:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = sorted(list(set([response["adapter_dir"] for response in responses])))

        total_service = get_full_service_over_time_fair(responses, T, window, x_ticks)
        total_throughput = get_full_throughput_over_time(responses, T, window, x_ticks)
        total_response_time = get_full_response_time_over_time(responses, T, window, x_ticks)
        #response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
        user_wasted_tokens, user_interaction_aborted, user_requests_aborted = get_wasted_tokens(responses)

        print(f"---------------{names[1]} Wasted tokens stats-----------")
        mean_wasted_tokens = 0
        percentile_5_wasted_tokens = 0
        percentile_95_wasted_tokens = 0
        #total_wasted_tokens = 0
        # for key, val in user_wasted_tokens.items():
        #     print(f"user: {key}, wasted_tokens: {val}")
        if len(user_wasted_tokens) > 0:
            wasted_tokens_array = np.array(list(user_wasted_tokens.values()))
            mean_wasted_tokens = np.mean(wasted_tokens_array)
            percentile_5_wasted_tokens = np.percentile(wasted_tokens_array, 5)
            percentile_95_wasted_tokens = np.percentile(wasted_tokens_array, 95)

            print(f"Mean of wasted tokens: {mean_wasted_tokens}")
            print(f"5th percentile of wasted tokens: {percentile_5_wasted_tokens}")
            print(f"95th percentile of wasted tokens: {percentile_95_wasted_tokens}")
        
        input2_wasted_tokens_stats = [mean_wasted_tokens, percentile_5_wasted_tokens, percentile_95_wasted_tokens]
        print(f"---------------{names[1]} aborted interaction stats-----------")
        mean_user_interaction_aborted = 0
        percentile_5_user_interaction_aborted = 0
        percentile_95_user_interaction_aborted = 0
        if len(user_interaction_aborted) > 0:
            user_int_aborted_vals = []
            for key, val in user_interaction_aborted.items():
                #print(f"user: {key}, len: {len(val)}")#, interaction_ids: {val}")
                user_int_aborted_vals.append(len(val))

            
            user_interaction_aborted_array = np.array(user_int_aborted_vals)
            mean_user_interaction_aborted = np.mean(user_interaction_aborted_array)
            percentile_5_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 5)
            percentile_95_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 95)

            print(f"Mean of user_interaction_aborted: {mean_user_interaction_aborted}")
            print(f"5th percentile of user_interaction_aborted: {percentile_5_user_interaction_aborted}")
            print(f"95th percentile of user_interaction_aborted: {percentile_95_user_interaction_aborted}")
        
        input2_aborted_interaction_stats = [mean_user_interaction_aborted, percentile_5_user_interaction_aborted, percentile_95_user_interaction_aborted]
        print(f"---------------{names[1]} aborted requests stats-----------")
        mean_user_requests_aborted = 0
        percentile_5_user_requests_aborted = 0
        percentile_95_user_requests_aborted = 0
        if len(user_requests_aborted) > 0:
            user_req_aborted_vals = []
            for key, val in user_requests_aborted.items():
                #print(f"user: {key},  len: {len(val)}")#, req_ids: {val}")
                user_req_aborted_vals.append(len(val))

            user_requests_aborted_array = np.array(user_req_aborted_vals)
            mean_user_requests_aborted = np.mean(user_requests_aborted_array)
            percentile_5_user_requests_aborted = np.percentile(user_requests_aborted_array, 5)
            percentile_95_user_requests_aborted = np.percentile(user_requests_aborted_array, 95)

            print(f"Mean of user_requests_aborted: {mean_user_requests_aborted}")
            print(f"5th percentile of user_requests_aborted: {percentile_5_user_requests_aborted}")
            print(f"95th percentile of user_requests_aborted: {percentile_95_user_requests_aborted}")
        input2_aborted_req_stats = [mean_user_requests_aborted, percentile_5_user_requests_aborted, percentile_95_user_requests_aborted]
    
    for exp in exps_3:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = sorted(list(set([response["adapter_dir"] for response in responses])))

        total_service = get_full_service_over_time_fair(responses, T, window, x_ticks)
        total_throughput = get_full_throughput_over_time(responses, T, window, x_ticks)
        total_response_time = get_full_response_time_over_time(responses, T, window, x_ticks)
        #response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
        user_wasted_tokens, user_interaction_aborted, user_requests_aborted = get_wasted_tokens(responses)

        print(f"---------------{names[2]} Wasted tokens stats-----------")
        mean_wasted_tokens = 0
        percentile_5_wasted_tokens = 0
        percentile_95_wasted_tokens = 0
        #total_wasted_tokens = 0
        # for key, val in user_wasted_tokens.items():
        #     print(f"user: {key}, wasted_tokens: {val}")
        if len(user_wasted_tokens) > 0:
            wasted_tokens_array = np.array(list(user_wasted_tokens.values()))
            mean_wasted_tokens = np.mean(wasted_tokens_array)
            percentile_5_wasted_tokens = np.percentile(wasted_tokens_array, 5)
            percentile_95_wasted_tokens = np.percentile(wasted_tokens_array, 95)

            print(f"Mean of wasted tokens: {mean_wasted_tokens}")
            print(f"5th percentile of wasted tokens: {percentile_5_wasted_tokens}")
            print(f"95th percentile of wasted tokens: {percentile_95_wasted_tokens}")
        
        input3_wasted_tokens_stats = [mean_wasted_tokens, percentile_5_wasted_tokens, percentile_95_wasted_tokens]
        print(f"---------------{names[2]} aborted interaction stats-----------")
        mean_user_interaction_aborted = 0
        percentile_5_user_interaction_aborted = 0
        percentile_95_user_interaction_aborted = 0
        if len(user_interaction_aborted) > 0:
            user_int_aborted_vals = []
            for key, val in user_interaction_aborted.items():
                #print(f"user: {key}, len: {len(val)}")#, interaction_ids: {val}")
                user_int_aborted_vals.append(len(val))

            
            user_interaction_aborted_array = np.array(user_int_aborted_vals)
            mean_user_interaction_aborted = np.mean(user_interaction_aborted_array)
            percentile_5_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 5)
            percentile_95_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 95)

            print(f"Mean of user_interaction_aborted: {mean_user_interaction_aborted}")
            print(f"5th percentile of user_interaction_aborted: {percentile_5_user_interaction_aborted}")
            print(f"95th percentile of user_interaction_aborted: {percentile_95_user_interaction_aborted}")
        
        input3_aborted_interaction_stats = [mean_user_interaction_aborted, percentile_5_user_interaction_aborted, percentile_95_user_interaction_aborted]
        print(f"---------------{names[2]} aborted requests stats-----------")
        mean_user_requests_aborted = 0
        percentile_5_user_requests_aborted = 0
        percentile_95_user_requests_aborted = 0
        if len(user_requests_aborted) > 0:
            user_req_aborted_vals = []
            for key, val in user_requests_aborted.items():
                #print(f"user: {key},  len: {len(val)}")#, req_ids: {val}")
                user_req_aborted_vals.append(len(val))

            user_requests_aborted_array = np.array(user_req_aborted_vals)
            mean_user_requests_aborted = np.mean(user_requests_aborted_array)
            percentile_5_user_requests_aborted = np.percentile(user_requests_aborted_array, 5)
            percentile_95_user_requests_aborted = np.percentile(user_requests_aborted_array, 95)

            print(f"Mean of user_requests_aborted: {mean_user_requests_aborted}")
            print(f"5th percentile of user_requests_aborted: {percentile_5_user_requests_aborted}")
            print(f"95th percentile of user_requests_aborted: {percentile_95_user_requests_aborted}")
        input3_aborted_req_stats = [mean_user_requests_aborted, percentile_5_user_requests_aborted, percentile_95_user_requests_aborted]

    for exp in exps_4:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = sorted(list(set([response["adapter_dir"] for response in responses])))

        total_service = get_full_service_over_time_fair(responses, T, window, x_ticks)
        total_throughput = get_full_throughput_over_time(responses, T, window, x_ticks)
        total_response_time = get_full_response_time_over_time(responses, T, window, x_ticks)
        #response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
        user_wasted_tokens, user_interaction_aborted, user_requests_aborted = get_wasted_tokens(responses)

        print(f"---------------{names[3]} Wasted tokens stats-----------")
        mean_wasted_tokens = 0
        percentile_5_wasted_tokens = 0
        percentile_95_wasted_tokens = 0
        #total_wasted_tokens = 0
        # for key, val in user_wasted_tokens.items():
        #     print(f"user: {key}, wasted_tokens: {val}")
        if len(user_wasted_tokens) > 0:
            wasted_tokens_array = np.array(list(user_wasted_tokens.values()))
            mean_wasted_tokens = np.mean(wasted_tokens_array)
            percentile_5_wasted_tokens = np.percentile(wasted_tokens_array, 5)
            percentile_95_wasted_tokens = np.percentile(wasted_tokens_array, 95)

            print(f"Mean of wasted tokens: {mean_wasted_tokens}")
            print(f"5th percentile of wasted tokens: {percentile_5_wasted_tokens}")
            print(f"95th percentile of wasted tokens: {percentile_95_wasted_tokens}")
        
        input4_wasted_tokens_stats = [mean_wasted_tokens, percentile_5_wasted_tokens, percentile_95_wasted_tokens]
        print(f"---------------{names[3]} aborted interaction stats-----------")
        mean_user_interaction_aborted = 0
        percentile_5_user_interaction_aborted = 0
        percentile_95_user_interaction_aborted = 0
        if len(user_interaction_aborted) > 0:
            user_int_aborted_vals = []
            for key, val in user_interaction_aborted.items():
                #print(f"user: {key}, len: {len(val)}")#, interaction_ids: {val}")
                user_int_aborted_vals.append(len(val))

            
            user_interaction_aborted_array = np.array(user_int_aborted_vals)
            mean_user_interaction_aborted = np.mean(user_interaction_aborted_array)
            percentile_5_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 5)
            percentile_95_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 95)

            print(f"Mean of user_interaction_aborted: {mean_user_interaction_aborted}")
            print(f"5th percentile of user_interaction_aborted: {percentile_5_user_interaction_aborted}")
            print(f"95th percentile of user_interaction_aborted: {percentile_95_user_interaction_aborted}")
        
        input4_aborted_interaction_stats = [mean_user_interaction_aborted, percentile_5_user_interaction_aborted, percentile_95_user_interaction_aborted]
        print(f"---------------{names[3]} aborted requests stats-----------")
        mean_user_requests_aborted = 0
        percentile_5_user_requests_aborted = 0
        percentile_95_user_requests_aborted = 0
        if len(user_requests_aborted) > 0:
            user_req_aborted_vals = []
            for key, val in user_requests_aborted.items():
                #print(f"user: {key},  len: {len(val)}")#, req_ids: {val}")
                user_req_aborted_vals.append(len(val))

            user_requests_aborted_array = np.array(user_req_aborted_vals)
            mean_user_requests_aborted = np.mean(user_requests_aborted_array)
            percentile_5_user_requests_aborted = np.percentile(user_requests_aborted_array, 5)
            percentile_95_user_requests_aborted = np.percentile(user_requests_aborted_array, 95)

            print(f"Mean of user_requests_aborted: {mean_user_requests_aborted}")
            print(f"5th percentile of user_requests_aborted: {percentile_5_user_requests_aborted}")
            print(f"95th percentile of user_requests_aborted: {percentile_95_user_requests_aborted}")
        input4_aborted_req_stats = [mean_user_requests_aborted, percentile_5_user_requests_aborted, percentile_95_user_requests_aborted]
    
    for exp in exps_5:
        config = exp["config"]
        result = exp["result"]

        responses = result["responses"]
        T = max([response["req_time"] for response in responses])
        T = int(T) / 10 * 10
        num_x = 100
        window = 60
        x_ticks = [T / num_x * i for i in range(num_x)]

        users = sorted(list(set([response["adapter_dir"] for response in responses])))

        total_service = get_full_service_over_time_fair(responses, T, window, x_ticks)
        total_throughput = get_full_throughput_over_time(responses, T, window, x_ticks)
        total_response_time = get_full_response_time_over_time(responses, T, window, x_ticks)
        #response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
        user_wasted_tokens, user_interaction_aborted, user_requests_aborted = get_wasted_tokens(responses)

        print(f"---------------{names[4]} Wasted tokens stats-----------")
        mean_wasted_tokens = 0
        percentile_5_wasted_tokens = 0
        percentile_95_wasted_tokens = 0
        #total_wasted_tokens = 0
        # for key, val in user_wasted_tokens.items():
        #     print(f"user: {key}, wasted_tokens: {val}")
        if len(user_wasted_tokens) > 0:
            wasted_tokens_array = np.array(list(user_wasted_tokens.values()))
            mean_wasted_tokens = np.mean(wasted_tokens_array)
            percentile_5_wasted_tokens = np.percentile(wasted_tokens_array, 5)
            percentile_95_wasted_tokens = np.percentile(wasted_tokens_array, 95)

            print(f"Mean of wasted tokens: {mean_wasted_tokens}")
            print(f"5th percentile of wasted tokens: {percentile_5_wasted_tokens}")
            print(f"95th percentile of wasted tokens: {percentile_95_wasted_tokens}")
        
        input5_wasted_tokens_stats = [mean_wasted_tokens, percentile_5_wasted_tokens, percentile_95_wasted_tokens]
        print(f"---------------{names[4]} aborted interaction stats-----------")
        mean_user_interaction_aborted = 0
        percentile_5_user_interaction_aborted = 0
        percentile_95_user_interaction_aborted = 0
        if len(user_interaction_aborted) > 0:
            user_int_aborted_vals = []
            for key, val in user_interaction_aborted.items():
                #print(f"user: {key}, len: {len(val)}")#, interaction_ids: {val}")
                user_int_aborted_vals.append(len(val))

            
            user_interaction_aborted_array = np.array(user_int_aborted_vals)
            mean_user_interaction_aborted = np.mean(user_interaction_aborted_array)
            percentile_5_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 5)
            percentile_95_user_interaction_aborted = np.percentile(user_interaction_aborted_array, 95)

            print(f"Mean of user_interaction_aborted: {mean_user_interaction_aborted}")
            print(f"5th percentile of user_interaction_aborted: {percentile_5_user_interaction_aborted}")
            print(f"95th percentile of user_interaction_aborted: {percentile_95_user_interaction_aborted}")
        
        input5_aborted_interaction_stats = [mean_user_interaction_aborted, percentile_5_user_interaction_aborted, percentile_95_user_interaction_aborted]
        print(f"---------------{names[4]} aborted requests stats-----------")
        mean_user_requests_aborted = 0
        percentile_5_user_requests_aborted = 0
        percentile_95_user_requests_aborted = 0
        if len(user_requests_aborted) > 0:
            user_req_aborted_vals = []
            for key, val in user_requests_aborted.items():
                #print(f"user: {key},  len: {len(val)}")#, req_ids: {val}")
                user_req_aborted_vals.append(len(val))

            user_requests_aborted_array = np.array(user_req_aborted_vals)
            mean_user_requests_aborted = np.mean(user_requests_aborted_array)
            percentile_5_user_requests_aborted = np.percentile(user_requests_aborted_array, 5)
            percentile_95_user_requests_aborted = np.percentile(user_requests_aborted_array, 95)

            print(f"Mean of user_requests_aborted: {mean_user_requests_aborted}")
            print(f"5th percentile of user_requests_aborted: {percentile_5_user_requests_aborted}")
            print(f"95th percentile of user_requests_aborted: {percentile_95_user_requests_aborted}")
        input5_aborted_req_stats = [mean_user_requests_aborted, percentile_5_user_requests_aborted, percentile_95_user_requests_aborted]

input1_wasted_tokens_error = [[input1_wasted_tokens_stats[0] - input1_wasted_tokens_stats[1]], [input1_wasted_tokens_stats[2] - input1_wasted_tokens_stats[0]]]
input1_aborted_interaction_error = [[input1_aborted_interaction_stats[0] - input1_aborted_interaction_stats[1]], [input1_aborted_interaction_stats[2] - input1_aborted_interaction_stats[0]]]
input1_aborted_requests_error = [[input1_aborted_req_stats[0] - input1_aborted_req_stats[1]], [input1_aborted_req_stats[2] - input1_aborted_req_stats[0]]]

input2_wasted_tokens_error = [[input2_wasted_tokens_stats[0] - input2_wasted_tokens_stats[1]], [input2_wasted_tokens_stats[2] - input2_wasted_tokens_stats[0]]]
input2_aborted_interaction_error = [[input2_aborted_interaction_stats[0] - input2_aborted_interaction_stats[1]], [input2_aborted_interaction_stats[2] - input2_aborted_interaction_stats[0]]]
input2_aborted_requests_error = [[input2_aborted_req_stats[0] - input2_aborted_req_stats[1]], [input2_aborted_req_stats[2] - input2_aborted_req_stats[0]]]

input3_wasted_tokens_error = [[input3_wasted_tokens_stats[0] - input3_wasted_tokens_stats[1]], [input3_wasted_tokens_stats[2] - input3_wasted_tokens_stats[0]]]
input3_aborted_interaction_error = [[input3_aborted_interaction_stats[0] - input3_aborted_interaction_stats[1]], [input3_aborted_interaction_stats[2] - input3_aborted_interaction_stats[0]]]
input3_aborted_requests_error = [[input3_aborted_req_stats[0] - input3_aborted_req_stats[1]], [input3_aborted_req_stats[2] - input3_aborted_req_stats[0]]]

input4_wasted_tokens_error = [[input4_wasted_tokens_stats[0] - input4_wasted_tokens_stats[1]], [input4_wasted_tokens_stats[2] - input4_wasted_tokens_stats[0]]]
input4_aborted_interaction_error = [[input4_aborted_interaction_stats[0] - input4_aborted_interaction_stats[1]], [input4_aborted_interaction_stats[2] - input4_aborted_interaction_stats[0]]]
input4_aborted_requests_error = [[input4_aborted_req_stats[0] - input4_aborted_req_stats[1]], [input4_aborted_req_stats[2] - input4_aborted_req_stats[0]]]

input5_wasted_tokens_error = [[input5_wasted_tokens_stats[0] - input5_wasted_tokens_stats[1]], [input5_wasted_tokens_stats[2] - input5_wasted_tokens_stats[0]]]
input5_aborted_interaction_error = [[input5_aborted_interaction_stats[0] - input5_aborted_interaction_stats[1]], [input5_aborted_interaction_stats[2] - input5_aborted_interaction_stats[0]]]
input5_aborted_requests_error = [[input5_aborted_req_stats[0] - input5_aborted_req_stats[1]], [input5_aborted_req_stats[2] - input5_aborted_req_stats[0]]]

hatch_pattern = [
    "/",    # Diagonal lines
    "\\",   # Backward diagonal lines
    "|",    # Vertical lines
    "-",    # Horizontal lines
    "+",    # Crossed lines
    "x",    # Crossed diagonal lines
    "o",    # Dots
    "O",    # Circular dots
    ".",    # Small dots
    "*",    # Stars
]
colors = [
    "#90ee90",  # Light Green
    "#add8e6",  # Light Blue
    "#ffb6c1",  # Light Pink
    "#e6e6fa",  # Lavender
    "#d3d3d3"   # Light Gray
    "#f08080",  # Light Coral
    "#ffffe0",  # Light Yellow
    "#87cefa",  # Light Sky Blue
    "#d3d3d3"   # Light Gray
]
plt.figure(figsize=(6, 6))
plt.bar(names[0], input1_wasted_tokens_stats[0], yerr=input1_wasted_tokens_error, capsize=10, label=names[0], hatch = hatch_pattern[0], color=colors[0])
plt.bar(names[1], input2_wasted_tokens_stats[0], yerr=input2_wasted_tokens_error, capsize=10, label=names[1], hatch = hatch_pattern[1], color=colors[1])
plt.bar(names[2], input3_wasted_tokens_stats[0], yerr=input3_wasted_tokens_error, capsize=10, label=names[2], hatch = hatch_pattern[2], color=colors[2])
plt.bar(names[3], input4_wasted_tokens_stats[0], yerr=input4_wasted_tokens_error, capsize=10, label=names[3], hatch = hatch_pattern[3], color=colors[3])
#plt.bar(names[4], input5_wasted_tokens_stats[0], yerr=input5_wasted_tokens_error, capsize=10, label=names[4], hatch = hatch_pattern[4], color=colors[4])
#plt.title('Wasted Tokens Stats')
plt.ylabel('Wasted tokens')
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
plt.legend()
plt.savefig("wasted_token_rpmvsfs.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 6))
plt.bar(names[0], input1_aborted_interaction_stats[0], yerr=input1_aborted_interaction_error, capsize=10, label=names[0], hatch = hatch_pattern[0], color=colors[0])
plt.bar(names[1], input2_aborted_interaction_stats[0], yerr=input2_aborted_interaction_error, capsize=10, label=names[1], hatch = hatch_pattern[1],  color=colors[1])
plt.bar(names[2], input3_aborted_interaction_stats[0], yerr=input3_aborted_interaction_error, capsize=10, label=names[2], hatch = hatch_pattern[2],  color=colors[2])
plt.bar(names[3], input4_aborted_interaction_stats[0], yerr=input4_aborted_interaction_error, capsize=10, label=names[3], hatch = hatch_pattern[3],  color=colors[3])
#plt.bar(names[4], input5_aborted_interaction_stats[0], yerr=input5_aborted_interaction_error, capsize=10, label=names[4], hatch = hatch_pattern[4],  color=colors[4])
#plt.title('Aborted Interaction Stats')
plt.ylabel('Aborted interactions')
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
plt.legend()
plt.savefig("aborted_interaction_rpmvsfs.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 6))
plt.bar(names[0], input1_aborted_req_stats[0], yerr=input1_aborted_requests_error, capsize=10, label=names[0], hatch = hatch_pattern[0], color=colors[0])
plt.bar(names[1], input2_aborted_req_stats[0], yerr=input2_aborted_requests_error, capsize=10, label=names[1], hatch = hatch_pattern[1],  color=colors[1])
plt.bar(names[2], input3_aborted_req_stats[0], yerr=input3_aborted_requests_error, capsize=10, label=names[2], hatch = hatch_pattern[2],  color=colors[2])
plt.bar(names[3], input4_aborted_req_stats[0], yerr=input4_aborted_requests_error, capsize=10, label=names[3], hatch = hatch_pattern[3],  color=colors[3])
#plt.bar(names[4], input5_aborted_req_stats[0], yerr=input5_aborted_requests_error, capsize=10, label=names[4], hatch = hatch_pattern[4],  color=colors[4])
#plt.title('Aborted Requests Stats')
plt.ylabel('Aborted requests')
plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
plt.legend()
plt.savefig("aborted_reqs_rpmvsfs.pdf", bbox_inches="tight")
plt.show()

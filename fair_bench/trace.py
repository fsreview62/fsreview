from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import pandas as pd
import os

class Request:
    def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time, sys_len=100, app=2, input99app=258, sys99app = 100, output99app=516, priorityfactor=1, llmcalls=3, app_limit = 10000, llmcalls_made = 0, interaction_id = 0):
        self.req_id = req_id
        self.model_dir = model_dir 
        self.adapter_dir = adapter_dir
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.req_time = req_time
        self.sys_len = sys_len
        self.app = app
        self.input99app = input99app
        self.sys99app = sys99app
        self.output99app = output99app
        self.priorityfactor = priorityfactor
        self.llmcalls = llmcalls
        self.app_limit = app_limit
        self.llmcalls_made = llmcalls_made
        self.interaction_id = interaction_id

    
    def __lt__(self, other):
        return self.req_time < other.req_time
    
    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, " \
               f"prompt_len={self.prompt_len}, output_len={self.output_len}, " \
               f"req_time={self.req_time}, " \
               f"sys_len={self.sys_len}, "\
               f"app={self.app}, " \
               f"input99app={self.input99app}, "\
               f"sys99app={self.sys99app}, "\
               f"output99app={self.output99app}, "\
               f"priorityfactor={self.priorityfactor}, "\
               f"llmcalls={self.llmcalls}, "\
               f"app_limit={self.app_limit}, "\
               f"llmcalls_made={self.llmcalls_made}, "\
               f"interaction_id={self.interaction_id}, "


def dummy_prompt(prompt_len):
    return "Hello " * prompt_len


def generate_requests_increase(num_adapters, alpha, req_rate, cv, duration,
                               input_range, output_range, on_off, mode,
                               adapter_dirs, # (base_dir, adapter_dir))
                               num_apps,
                               seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    requests = []
    # generate for adapter 0
    tot_req = int(req_rate[0] * duration)
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
    tic = 0
    for i in range(tot_req):
        tic += 1 / req_rate[0]
        if on_off != -1 and int(tic // on_off) & 1 == 1:
            continue
        requests.append(Request(len(requests), adapter_dirs[0][0], adapter_dirs[0][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    # generate for adapter 1
    start_rate = 0.2
    end_rate = req_rate[1]
    tot_req = int(0.5 * (start_rate + end_rate) * duration)
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
    tic = 0
    for i in range(tot_req):
        current_rate = start_rate + (end_rate - start_rate) * tic / duration
        tic += 1 / current_rate
        requests.append(Request(len(requests), adapter_dirs[1][0], adapter_dirs[1][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))

    requests = sorted(requests)
    return requests

def divide_reqs(tot_users):

    #pass
    fp = os.path.expanduser('~/sampled_10k.csv')
    df = pd.read_csv(fp)
    # user1 = df[(df['app'] == 1) & (df['prompt_len'] < 2048)]
    # user2 = df[(df['app'] == 2) & (df['prompt_len'] < 2048)]

    # user1reqs = user1.head(100)
    # user2reqs = user2.head(100)

    # print(f"user 1 reqs satisfying the prompt len condition: {len(user1reqs)}")
    # print(f"user 2 reqs satisfying the prompt len condition: {len(user2reqs)}")
    
    userreqs = []
    # userreqs.append(user1reqs)
    # userreqs.append(user2reqs)

    # Define number of users and maximum rows per user
    #tot_users = 2
    users = [i for i in range(tot_users)]  # You can add more users
    max_reqs_per_user = 100

    # Dictionary to store the DataFrame slices for each user
    user_dfs = {}

    # Iterate through users and assign chunks of rows
    for i, user in enumerate(users):
        start_row = i * max_reqs_per_user
        end_row = start_row + max_reqs_per_user
        user_dfs[user] =  df[(df['prompt_len'] < 2048)].iloc[start_row:end_row] #df[start_row:end_row]
    
    for user in user_dfs.keys():
        userreqs.append(user_dfs[user])
    return userreqs

def generate_requests_uniform_real(num_adapters, alpha, req_rate, cv, duration,
                              input_range, output_range, on_off, mode,
                              adapter_dirs, # (base_dir, adapter_dir))
                              num_apps,
                              seed=42):
    assert num_adapters == len(req_rate)
    userreqs = divide_reqs(num_adapters)

    np.random.seed(seed)

    requests = []
    int_max_id = 0
    #client_req = {adapter_dirs[i]: [] for i in range(num_adapters)}
    for i in range(num_adapters):
        #tot_req = int(req_rate[i] * duration)
        tot_req = int(len(userreqs[i]))
        #tot_req = 12
        user_req_df = userreqs[i]
        #input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        input_lens = user_req_df['prompt_len'].to_numpy()
        output_lens = user_req_df['output_len'].to_numpy()
        app_lens = user_req_df['app'].to_numpy()

        #llmcalls_map = user_req_df.groupby('app')['llmcalls'].apply(list).to_dict()
        llmcalls_map = user_req_df.groupby('app')['llmcalls'].apply(lambda x: list(set(x))).to_dict()
        num_apps = len(llmcalls_map)
        print(f"num_apps for user: {i}: {num_apps}, {llmcalls_map}")

        interaction_id = int_max_id
        interaction_id_list = []

        # Dictionary to keep track of requests count per app
        request_count = {app: 0 for app in np.unique(app_lens)}
        interaction_app_track = {app: [] for app in llmcalls_map.keys()}
        print(f"interaction_app_track: {interaction_app_track}")

        # Iterate through app_lens
        for x, app in enumerate(app_lens):
            # Increment request count for the current app
            request_count[app] += 1

            if llmcalls_map[app][0] != 1:
                if request_count[app] == 1:
                    interaction_id+=1
                    interaction_app_track[app].append(interaction_id)
                    interaction_id_list.append(interaction_id)
                elif request_count[app] % llmcalls_map[app][0] == 1:
                    interaction_id+=1
                    interaction_app_track[app].append(interaction_id)
                    interaction_id_list.append(interaction_id)
                else:
                    interaction_id_list.append(interaction_app_track[app][-1])
            else:
                interaction_id+=1
                interaction_app_track[app].append(interaction_id)
                interaction_id_list.append(interaction_id)

        int_max_id = interaction_id 

        sys_lens = user_req_df['syslen'].to_numpy()
        input99app = user_req_df['input99app'].to_numpy()
        sys99app = user_req_df['sys99app'].to_numpy()
        output99app = user_req_df['output99app'].to_numpy()
        priorityfactor = [1]*len(input99app)
        llmcalls = [] #user_req_df['llmcalls'].to_numpy()
        llmcalls_made = []
        app_limit = [200]*len(input99app)
        #int_lens = []

        llmcalls_made_track = {l:[] for l in llmcalls_map.keys()}

        for j in range(len(app_lens)):
            app_select = app_lens[j]
            llmcalls.append(llmcalls_map[app_select][0])
            if len(llmcalls_made_track[app_select]) == 0 or llmcalls_made_track[app_select][-1] == llmcalls[-1]:
                llmcalls_made.append(1)
                llmcalls_made_track[app_select].append(1)
            elif llmcalls[-1] == llmcalls_made_track[app_select][-1] + 1:
                llmcalls_made.append(llmcalls[-1])
                llmcalls_made_track[app_select].append(llmcalls[-1])
            else:
                llmcalls_made.append(llmcalls_made_track[app_select][-1] + 1)
                llmcalls_made_track[app_select].append(llmcalls_made_track[app_select][-1] + 1)
            print(f"inside app_select: {app_select}, llmcalls_made[-1]: {llmcalls_made[-1]}, llmcalls: {llmcalls[-1]}")

        tic = np.random.rand() * 1 / req_rate[i]
        req_id = tot_req * i
        print(f"req_id: {req_id}")
        #int_id = 0
        for j in range(tot_req):
            tic += 1 / req_rate[i] # assume req rate 0.5, then tic = 2
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            # requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
            #                         dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
            #                         tic, int(sys_lens[j]), int(app_lens[j]), int(input99app[j]), int(sys99app[j]), int(output99app[j]), int(priorityfactor[j]), int(llmcalls[j]), int(app_limit[j]))) # first req at 2 sec, then next one at 4 sec as tic increases by 2 for every req.
            requests.append(Request(int(req_id), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic, int(sys_lens[j]), int(app_lens[j]), int(input99app[j]), int(sys99app[j]), int(output99app[j]), int(priorityfactor[j]), int(llmcalls[j]), int(app_limit[j]), int(llmcalls_made[j]), int(interaction_id_list[j]))) # first req at 2 sec, then next one at 4 sec as tic increases by 2 for every req.
            req_id+=1
    print(f"requests formed: {len(requests)}")
    #print(f"requests: {requests}")
    requests = sorted(requests)

    print("")
    return requests

def generate_requests_uniform(num_adapters, alpha, req_rate, cv, duration,
                              input_range, output_range, on_off, mode,
                              adapter_dirs, # (base_dir, adapter_dir))
                              num_apps,
                              seed=42):
    assert num_adapters == len(req_rate)
    np.random.seed(seed)

    requests = []
    int_max_id = 0
    #client_req = {adapter_dirs[i]: [] for i in range(num_adapters)}
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration)
        #tot_req = 12
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        app_lens = np.random.randint(1,num_apps+1, tot_req)
        #app_lens = np.array([1,2,1,2,1,2,1,2,1,1,2,2])
        if i == 0:
            app_lens = np.random.randint(1,2,tot_req)
        else:
            app_lens = np.random.randint(2,3,tot_req)

        interaction_id = int_max_id
        interaction_id_list = []

        # Dictionary to keep track of requests count per app
        request_count = {app: 0 for app in np.unique(app_lens)}
        interaction_app_track = {app: [] for app in np.unique(app_lens)}

        # Iterate through app_lens
        for x, app in enumerate(app_lens):
            # Increment request count for the current app
            request_count[app] += 1

            if request_count[app] == 1:
                interaction_id+=1
                interaction_app_track[app].append(interaction_id)
                interaction_id_list.append(interaction_id)
            elif request_count[app] % 3 == 1:
                interaction_id+=1
                interaction_app_track[app].append(interaction_id)
                interaction_id_list.append(interaction_id)
            else:
                interaction_id_list.append(interaction_app_track[app][-1])

        int_max_id = interaction_id 

        sys_lens = []
        input99app = []
        sys99app = []
        output99app = []
        priorityfactor = []
        llmcalls = []
        llmcalls_made = []
        app_limit = []
        int_lens = []

        llmcalls_made_track = {l:[] for l in range(1, num_apps+1)}

        for j in range(len(app_lens)):
            app_select = app_lens[j] % (num_apps + 1)

            if app_select == 1:
                input99app.append(50)
                sys99app.append(100)
                output99app.append(258)
                sys_lens.append(10)
                priorityfactor.append(1)
                llmcalls.append(3)
                if len(llmcalls_made_track[app_select]) == 0 or llmcalls_made_track[app_select][-1] == llmcalls[-1]:
                    llmcalls_made.append(1)
                    llmcalls_made_track[app_select].append(1)
                    # int_id+=1
                    # int_lens.append(int_id)
                elif llmcalls[-1] == llmcalls_made_track[app_select][-1] + 1:
                    llmcalls_made.append(llmcalls[-1])
                    llmcalls_made_track[app_select].append(llmcalls[-1])
                    #int_lens.append(int_id)
                else:
                    llmcalls_made.append(llmcalls_made_track[app_select][-1] + 1)
                    llmcalls_made_track[app_select].append(llmcalls_made_track[app_select][-1] + 1)
                    #int_lens.append(int_id)
                #llmcalls_made.append(j%3)
                print(f"inside app_select: {app_select}, llmcalls_made[-1]: {llmcalls_made[-1]}, llmcalls: {llmcalls[-1]}")
                app_limit.append(200)
            elif app_select == 2:
                input99app.append(580)
                sys99app.append(2000)
                output99app.append(258)
                sys_lens.append(16)
                priorityfactor.append(1)
                llmcalls.append(3)
                if len(llmcalls_made_track[app_select]) == 0 or llmcalls_made_track[app_select][-1] == llmcalls[-1]:
                    llmcalls_made.append(1)
                    llmcalls_made_track[app_select].append(1)
                elif llmcalls[-1] == llmcalls_made_track[app_select][-1] + 1:
                    llmcalls_made.append(llmcalls[-1])
                    llmcalls_made_track[app_select].append(llmcalls[-1])
                else:
                    llmcalls_made.append(llmcalls_made_track[app_select][-1] + 1)
                    llmcalls_made_track[app_select].append(llmcalls_made_track[app_select][-1] + 1)
                
                print(f"inside app_select: {app_select}, llmcalls_made[-1]: {llmcalls_made[-1]}, llmcalls: {llmcalls[-1]}")
                app_limit.append(200)
            elif app_select == 3:
                input99app.append(516)
                sys99app.append(880)
                output99app.append(100)
                sys_lens.append(88)
                priorityfactor.append(1)
                llmcalls.append(3)
                if len(llmcalls_made_track[app_select]) == 0 or llmcalls_made_track[app_select][-1] == llmcalls[-1]:
                    llmcalls_made.append(1)
                    llmcalls_made_track[app_select].append(1)
                elif llmcalls[-1] == llmcalls_made_track[app_select][-1] + 1:
                    llmcalls_made.append(llmcalls[-1])
                    llmcalls_made_track[app_select].append(llmcalls[-1])
                else:
                    llmcalls_made.append(llmcalls_made_track[app_select][-1] + 1)
                    llmcalls_made_track[app_select].append(llmcalls_made_track[app_select][-1] + 1)
                print(f"inside app_select: {app_select}, llmcalls_made[-1]: {llmcalls_made[-1]}, llmcalls: {llmcalls[-1]}")
                app_limit.append(200)
            elif app_select == 4:
                input99app.append(258)
                sys99app.append(1500)
                output99app.append(258)
                sys_lens.append(150)
                priorityfactor.append(1)
                llmcalls.append(3)
                if len(llmcalls_made_track[app_select]) == 0 or llmcalls_made_track[app_select][-1] == llmcalls[-1]:
                    llmcalls_made.append(1)
                    llmcalls_made_track[app_select].append(1)
                elif llmcalls[-1] == llmcalls_made_track[app_select][-1] + 1:
                    llmcalls_made.append(llmcalls[-1])
                    llmcalls_made_track[app_select].append(llmcalls[-1])
                else:
                    llmcalls_made.append(llmcalls_made_track[app_select][-1] + 1)
                    llmcalls_made_track[app_select].append(llmcalls_made_track[app_select][-1] + 1)
                print(f"inside app_select: {app_select}, llmcalls_made[-1]: {llmcalls_made[-1]}, llmcalls: {llmcalls[-1]}")
                app_limit.append(200)
            elif app_select == 5:
                input99app.append(1024)
                sys99app.append(1200)
                output99app.append(258)
                sys_lens.append(120)
                priorityfactor.append(1)
                llmcalls.append(3)
                if len(llmcalls_made_track[app_select]) == 0 or llmcalls_made_track[app_select][-1] == llmcalls[-1]:
                    llmcalls_made.append(1)
                    llmcalls_made_track[app_select].append(1)
                elif llmcalls[-1] == llmcalls_made_track[app_select][-1] + 1:
                    llmcalls_made.append(llmcalls[-1])
                    llmcalls_made_track[app_select].append(llmcalls[-1])
                else:
                    llmcalls_made.append(llmcalls_made_track[app_select][-1] + 1)
                    llmcalls_made_track[app_select].append(llmcalls_made_track[app_select][-1] + 1)
                print(f"inside app_select: {app_select}, llmcalls_made[-1]: {llmcalls_made[-1]}, llmcalls: {llmcalls[-1]}")
                app_limit.append(200)
            else:
                input99app.append(774)
                sys99app.append(1600)
                output99app.append(258)
                sys_lens.append(120)
                priorityfactor.append(1)
                llmcalls.append(3)
                llmcalls_made.append(j%3)
                print(f"inside other app_select: {app_select}, llmcalls_made[-1]: {llmcalls_made[-1]}, llmcalls: {llmcalls[-1]}")
                app_limit.append(200)
        
        # give input99app, sys99app, out99app

        tic = np.random.rand() * 1 / req_rate[i]
        req_id = tot_req * i
        print(f"req_id: {req_id}")
        #int_id = 0
        for j in range(tot_req):
            tic += 1 / req_rate[i] # assume req rate 0.5, then tic = 2
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            # requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
            #                         dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
            #                         tic, int(sys_lens[j]), int(app_lens[j]), int(input99app[j]), int(sys99app[j]), int(output99app[j]), int(priorityfactor[j]), int(llmcalls[j]), int(app_limit[j]))) # first req at 2 sec, then next one at 4 sec as tic increases by 2 for every req.
            requests.append(Request(int(req_id), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic, int(sys_lens[j]), int(app_lens[j]), int(input99app[j]), int(sys99app[j]), int(output99app[j]), int(priorityfactor[j]), int(llmcalls[j]), int(app_limit[j]), int(llmcalls_made[j]), int(interaction_id_list[j]))) # first req at 2 sec, then next one at 4 sec as tic increases by 2 for every req.
            # if len(requests) % 3 == 0:
            #     int_id+=1
            # if requests[-1].llmcalls == requests[-1].llmcalls_made:
            #     int_id+=1
            req_id+=1
    print(f"requests formed: {len(requests)}")
    #print(f"requests: {requests}")
    requests = sorted(requests)

    print("")
    return requests


def generate_requests_poisson_short_long(num_adapters, alpha, req_rate, cv, duration,
                                         input_range, output_range, on_off, mode,
                                         adapter_dirs, # (base_dir, adapter_dir))
                                         num_apps,
                                         seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    tot_req = int(sum(req_rate) * duration)

    # generate adapter id
    probs = np.random.rand(tot_req)
    ind = (probs > (req_rate[0] / (req_rate[0] + req_rate[1]))).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / sum(req_rate)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        if ind[i] == 0:
            if on_off != -1 and int(tic // on_off) & 1 == 1:
                continue
            input_len = input_lens[i] // 4
            output_len = output_lens[i] // 4
        else:
            input_len = input_lens[i]
            output_len = output_lens[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_len), int(input_len), int(output_len),
                                tic))
    return requests


def generate_requests_poisson_short_long_2(num_adapters, alpha, req_rate, cv, duration,
                                           input_range, output_range, on_off, mode,
                                           adapter_dirs, # (base_dir, adapter_dir))
                                           num_apps,
                                           seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    tot_req = int(sum(req_rate) * duration)

    # generate adapter id
    probs = np.random.rand(tot_req)
    ind = (probs > (req_rate[0] / (req_rate[0] + req_rate[1]))).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / sum(req_rate)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        if ind[i] == 0:
            if on_off != -1 and int(tic // on_off) & 1 == 1:
                continue
            input_len = output_lens[i]
            output_len = input_lens[i]
        else:
            input_len = input_lens[i]
            output_len = output_lens[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_len), int(input_len), int(output_len),
                                tic))
    return requests


def generate_requests_poisson(num_adapters, alpha, req_rate, cv, duration,
                              input_range, output_range, on_off, mode,
                              adapter_dirs, # (base_dir, adapter_dir))
                              num_apps,
                              seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    tot_req = int(sum(req_rate) * duration)

    # generate adapter id
    probs = np.random.rand(tot_req)
    ind = (probs > (req_rate[0] / (req_rate[0] + req_rate[1]))).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / sum(req_rate)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        if ind[i] == 0 and on_off != -1 and int(tic // on_off) & 1 == 1:
            continue
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    return requests


def generate_requests_dist_shift(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, on_off, mode,
                                 adapter_dirs, # (base_dir, adapter_dir))
                                 num_apps,
                                 seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    assert req_rate == [-1, -1]
    np.random.seed(seed)

    requests = []

    # on_off phase
    req_rate = [0.5, 2]
    on_off = 60
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration / 3)
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        tic = np.random.rand() * 1 / req_rate[i]
        for j in range(tot_req):
            tic += 1 / req_rate[i]
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic))

    # overload phase
    req_rate = [1, 1]
    on_off = -1
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration / 3)
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        tic = duration / 3 + np.random.rand() * 1 / req_rate[i]
        for j in range(tot_req):
            tic += 1 / req_rate[i]
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic))

    # proportional phase
    req_rate = [0.5, 1.5]
    on_off = -1
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration / 3)
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        tic = duration / 3 * 2 + np.random.rand() * 1 / req_rate[i]
        for j in range(tot_req):
            tic += 1 / req_rate[i]
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic))

    requests = sorted(requests)
    return requests


def generate_requests(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range, on_off, mode,
                      adapter_dirs, # (base_dir, adapter_dir)
                      num_apps,
                      seed=42):

    # for the paper suite, any of the mode's will be accessed.
    #num of requests determined by req_rate (req/sec) *duration (sec).
    # for 0.5 req/sec, and duration 60 *10 = 600 sec, 0.5*600 = 300 requests will be produced for 600sec or 10 min. Each min 0.5*60 = 30 reqs
    if mode == "increase":
        return generate_requests_increase(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs,num_apps,seed)
    elif mode == "uniform":
        return generate_requests_uniform(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs,num_apps,seed)
    elif mode == "uniform-real":
        return generate_requests_uniform_real(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs,num_apps,seed)
    elif mode == "poisson-short-long":
        return generate_requests_poisson_short_long(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs,num_apps,seed)
    elif mode == "poisson-short-long-2":
        return generate_requests_poisson_short_long_2(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, num_apps, seed)
    elif mode == "poisson":
        return generate_requests_poisson(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, num_apps, seed)
    elif mode == "dist_shift":
        return generate_requests_dist_shift(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, num_apps, seed)

    
    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    return requests


def get_real_requests(trace_file, req_rate, duration, base_model, adapter_dirs, input_range, output_range, seed=42):
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    conversations = downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range)
    model_mapping = generate_model_mapping(conversations, adapter_dirs)
    conversations = sort_and_rescale_by_req_time(conversations, duration)
    reqs = parse_into_req(base_model, conversations, model_mapping, tokenizer)
    return list(model_mapping.values()), reqs

def get_trace_data_m365(base_model, adapter_dirs, input, app_limit, app_stats="avg"):
    df: pd.DataFrame = pd.read_csv(input)
    print("Number of requests:", len(df))
    num_adapters = df["UserID"].nunique()
    user_ids = list(df["UserID"].unique())
    df["adapter_dir"] = df["UserID"].apply(lambda x: user_ids.index(x))
    reqs = []
    num_ranks = [0] * len(adapter_dirs)
    for idx, row in df.iterrows():
        adapter_dir = random.choice(adapter_dirs)
        p_len = max(1, int((row["PromptTokens"] + 7) // 8))
        d_len = max(1, int((row["DecodeTokens"] + 7) // 8))
        if (p_len + d_len) >= 2000:
            continue
        try:
            llmcalls_made = int(row["llmcalls_made"]) 
            llmcalls = int(row["llmcalls"])
        except:
            llmcalls_made = 0
            llmcalls = 1
        req_here = Request(
            req_id=int(idx),
            model_dir=base_model,
            adapter_dir=f"{adapter_dir}-{row['adapter_dir']}",
            prompt_len=p_len,
            output_len=d_len,
            req_time=float(row["ArrivalTime"]),
            app=row["app"],
            prompt=dummy_prompt(p_len),
            sys99app=max(1, int(row[f"sys_{app_stats}_app"])),
            input99app=max(1, int(row[f"input_{app_stats}_app"])),
            output99app=max(1, int(row[f"output_{app_stats}_app"])),
            sys_len=0,
            llmcalls=llmcalls,
            llmcalls_made=llmcalls_made + 1,
            interaction_id=row["interaction_id"],
            app_limit=app_limit,
        )
        num_ranks[adapter_dirs.index(adapter_dir)] += 1
        reqs.append(req_here)
    print(f"Num ranks: {num_ranks}")
    return reqs, num_adapters



# functions below are used to generate real requests
def downsample(json_file, req_rate, duration, tokenizer, input_range, output_range):
    with open(json_file, "r") as file:
       all_conversations = json.load(file)
    
    more_ratio = 2
    need_num = int(req_rate * duration)
    # sample a bit more than needed
    selected_indicies = np.random.choice(len(all_conversations), more_ratio * need_num, replace=False)
    downsampled_conversations = [all_conversations[idx] for idx in selected_indicies]
    for idx in reversed(range(len(downsampled_conversations))):
        conv = downsampled_conversations[idx]
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        if prompt_len >= input_range[1] or output_len >= output_range[1]:
            downsampled_conversations.pop(idx)
        
    downsampled_conversations = downsampled_conversations[:need_num]
    print(f"Downsampled {len(downsampled_conversations)}")
    return downsampled_conversations 

def generate_model_mapping(conversations, adapter_dirs):
    model_mapping = {}
    num_ranks = [0] * len(adapter_dirs)
    for conv in conversations:
        model = conv["model"]
        if model not in model_mapping.keys():
            adapter_dir = random.choice(adapter_dirs)
            name = f"{adapter_dir}-{num_ranks[adapter_dirs.index(adapter_dir)]}"
            num_ranks[adapter_dirs.index(adapter_dir)] += 1
            model_mapping[model] = name
    print(model_mapping)
    return model_mapping


def sort_and_rescale_by_req_time(conversations, duration):
    # sort first
    sorted_conversations = sorted(conversations, key=lambda d: d['tstamp']) 
    interval_start = sorted_conversations[0]["tstamp"]
    interval_end = sorted_conversations[-1]["tstamp"]
    # print(f"sorted time step: {[s['tstamp'] for s in sorted_conversations]}")

    for conv in conversations:
        tstamp = conv["tstamp"]
        assert interval_start <= tstamp and tstamp <= interval_end
        rescaled_tstamp = (tstamp - interval_start) / (interval_end - interval_start) * duration
        conv["tstamp"] = rescaled_tstamp
    return sorted_conversations 


def parse_into_req(base_model, conversations, model_mapping, tokenizer):
    reqs = []
    for idx, conv in enumerate(conversations):
        model = conv["model"]
        name = model_mapping[model]
        # print(conv["conversation"][0]["content"])
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        
        req = Request(req_id=idx, model_dir=base_model, adapter_dir=name, 
              prompt=dummy_prompt(prompt_len), prompt_len=prompt_len,
              output_len=output_len, req_time=conv["tstamp"])
        reqs.append(req)
    # print(reqs)
    return reqs


import asyncio
import uuid
from collections import deque
from typing import List, Optional
import time

import numpy as np

from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
from slora.server.router.req_queue import ReqQueue
from slora.utils.metric import attainment_func


class FSReqQueueOverloadLimit(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 adapter_dirs, fair_weights, cost_func, rate_limit=180,
                 input_price=1, output_price=2) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.input_price = input_price
        self.output_price = output_price
        self.sys_price = 1
        self.wosys = 1
        self.wosysvtc = 0 
        self.served = {}
        self.servedwosys = {}
        self.user_req_list = {}
        self.user_req_abort_list = {}
        self.user_req_abort_list_time = {}
        self.app_req_abort_list = {}
        self.app_req_abort_list_time = {}


        self.adapter_dirs = adapter_dirs
        self.fair_weights = fair_weights
        self.cost_func = cost_func
        self.rate_limit = rate_limit

        self.fairw = {}
        self.systokens = {}
        self.input99percuser = {}
        self.sys99percuser = {}
        self.output99percuser = {}
        self.priorityfactoruser = {}

        self.abort_req_list: List[str] = []
        #self.abort
        self.req_time_stamp = []
        self.init_bs = 1
        self.apprx_req_rate = 1
        self.apprx_bs = self.init_bs
        self.last_req_num = 0
        self.last_time = time.time()
        assert rate_limit is not None, "Please specify the rate limit for FairServe scheduler"
        self.rate_limit = rate_limit # per minute
        self.all_req_time_stamp = {}
        self.all_req_time_stamp_app = {}
        self.all_req_user_id = {}
        self.all_req_app_id = {}
        self.total_aborted = {}
        self.total_aborted_app = {}
        
        self.llmagent = [1,10,4,3,1,5,6,7,9,10,5]
        self.tokenperagent = [10,20,22,50,120,32,29,28,16,13,21]
        self.systoken_len = [10,200,88,150,120,160,174,196,144,130,105]
        #self.input99perc = [496,116,512,512,512,512,512,512,512,512,512]
        self.input99perc = [50,580,512,512,512,512,512,512,512,512,512]
        #self.output99perc = [512,512,512,512,512,512,512,512,512,512,512]
        self.output99perc = [258,258,512,512,512,512,512,512,512,512,512]
        self.sys99perc = [100,2000,880,1500,1200,1600,1740,1960,1440,1300,1050]
        self.priorityfactor = [1,1,2,5,1,1,1,1,1,1,1]

        for i in range(len(adapter_dirs)):
            if i < len(fair_weights):
                self.fairw[adapter_dirs[i]] = fair_weights[i]
            else:
                self.fairw[adapter_dirs[i]] = 1
        
        #print(len(adapter_dirs))
        self.adapter_dirs = sorted(self.adapter_dirs)
        for i in range(len(adapter_dirs)):
            #print(i)
            self.systokens[adapter_dirs[i]] = int(self.llmagent[i%10] * self.tokenperagent[i%10])
            self.input99percuser[adapter_dirs[i]] = int(self.input99perc[i%10])
            self.sys99percuser[adapter_dirs[i]] = int(self.sys99perc[i%10])
            self.output99percuser[adapter_dirs[i]] = int(self.output99perc[i%10])
            self.priorityfactoruser[adapter_dirs[i]] = int(self.priorityfactor[i%10])
        
        
    def append(self, req):
        cur_req_time = time.time()
        self.waiting_req_list.insert(0,req)
        # if self.check_past_one_minute_user(req.adapter_dir, cur_req_time):
        #     req.aborted = True
        #     print(f"aborted {req.request_id}")
        #     self.abort_req_list.append(req.request_id)
        #     if req.adapter_dir not in self.user_req_abort_list:
        #         self.user_req_abort_list[req.adapter_dir] = deque([req])
        #         self.user_req_abort_list_time[req.adapter_dir] = deque([cur_req_time])
        #     else:
        #         self.user_req_abort_list[req.adapter_dir].append(req)
        #         self.user_req_abort_list_time[req.adapter_dir].append(cur_req_time)
        #     #self.user_req_abort_list[]
        #     # aborted_count += 1
        #     # self.abort_req_list.append(req.request_id)
        #     return
        # else:
        #     req_app = req.app
        #     if self.check_past_one_minute_app(req_app, cur_req_time, req):
        #         req.aborted = True
        #         print(f"aborted by app {req.request_id}")
        #         self.abort_req_list.append(req.request_id)
        #         if req.app not in self.app_req_abort_list:
        #             self.app_req_abort_list[req.app] = deque([req])
        #             self.app_req_abort_list_time[req.app] = deque([cur_req_time])
        #         else:
        #             self.app_req_abort_list[req.app].append(req)
        #             self.app_req_abort_list_time[req.app].append(cur_req_time)
        #         return


        #self.waiting_req_list.insert(req)
        self.req_time_stamp.insert(0,cur_req_time)
        if req.adapter_dir not in self.all_req_time_stamp.keys():
            self.all_req_time_stamp[req.adapter_dir] = []
        if req.app not in self.all_req_time_stamp_app.keys():
            self.all_req_time_stamp_app[req.app] = []
        self.all_req_time_stamp_app[req.app].insert(0,cur_req_time)
        # self.all_req_app_id[req.app].append(req)
        # self.all_req_user_id[req.adapter_dir].append(req)
        self.all_req_time_stamp[req.adapter_dir].insert(0,cur_req_time)
        assert len(self.waiting_req_list) == len(self.req_time_stamp)

        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
            self.served[req.adapter_dir] = 0
            self.servedwosys[req.adapter_dir] = 0
        else:
            self.user_req_list[req.adapter_dir].append(req)

        # waiting queue was empty before
        if len(self.user_req_list[req.adapter_dir]) == 1:
            #return
            # lift counter
            cnts = [v for k, v in self.served.items()
                      if (len(self.user_req_list[k]) > 0 and k != req.adapter_dir)]
            if self.wosys == 1:
                cnts = [v for k, v in self.servedwosys.items()
                        if (len(self.user_req_list[k]) > 0 and k != req.adapter_dir)]
            if len(cnts) > 0:
                self.served[req.adapter_dir] = max(self.served[req.adapter_dir], min(cnts))
                if self.wosys == 1:
                    self.servedwosys[req.adapter_dir] = max(self.servedwosys[req.adapter_dir], min(cnts))

    
    def check_past_one_minute_user(self, adapter, check_time):
        counter = 0
        check_time = self.all_req_time_stamp[adapter][len(self.all_req_time_stamp[adapter])-1]
        if adapter not in self.all_req_time_stamp.keys():
            return False
        for _, req_time in enumerate(self.all_req_time_stamp[adapter]):
            # print(check_time - req_time)
            if req_time - check_time <= 60: # submitted in the last minute
                counter += 1
            if counter >= self.rate_limit:
                if adapter not in self.total_aborted.keys():
                    self.total_aborted[adapter] = 0
                self.total_aborted[adapter] += 1
                #print(f"client: {adapter} aborted {self.total_aborted[adapter]} / {len(self.all_req_time_stamp[adapter])}")
                return True
                
        return False
    def check_past_one_minute_app(self, req_app, check_time, req):
        counter = 0
        check_time = self.all_req_time_stamp_app[req_app][len(self.all_req_time_stamp_app[req_app])-1]
        if req_app not in self.all_req_time_stamp_app.keys():
            return False
        for _, req_time in enumerate(self.all_req_time_stamp_app[req_app]):
            # print(check_time - req_time)
            if req_time - check_time <= 60: # submitted in the last minute
                counter += 1
            if counter >= req.app_limit:
                if req_app not in self.total_aborted_app.keys():
                    self.total_aborted_app[req_app] = 0
                self.total_aborted_app[req_app] += 1
                #print(f"App: {req_app} aborted {self.total_aborted_app[req_app]} / {len(self.all_req_time_stamp_app[req_app])}")
                return True
                
        return False

    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                #print(f"init cache: req:{req}")
                #print(f"init cache: req.output_ids: {req.output_ids} len(req.output_ids): {len(req.output_ids)}") #a bunch of numbers come up.
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                           req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list]) # how much remaining output can be generated. 0th place of the array has the highest as cache list was sorted.
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array) # estimate of how much input can be in cache at the moment based on max of this array.
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max() #need to understand logic behind this 
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False

    def reset_abort_list(self):
        self.abort_req_list = []

    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        #print(f"generate_new_batch. lora_ranks {lora_ranks}")
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        if len(self.served) == 0:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        active_served = {k: v for k, v in self.served.items()}
        if self.wosys == 1:
            #print(f"active_served from wosys.")
            active_served = {k: v for k, v in self.servedwosys.items()}
        while True:
            if len(active_served) == 0:
                break
            adapter_dir = min(active_served, key=active_served.get) # need a new kv list here. One that omits the system tokens. 
            # for key, val in active_served.items():
            #     print(f"user: {key}, received service: {val}, num of requests in user queue: {len(self.user_req_list[key])}, selected user: {adapter_dir}, selected user service: {active_served[adapter_dir]}, selected user requests: {len(self.user_req_list[adapter_dir])}")
            # print(f"-----------------------------------------------------------------------------------------------------------------")
            
            #print(f"")
            if len(self.user_req_list[adapter_dir]) > 0:
                #print(f"active_served: {active_served}, min adapter_dir: {adapter_dir}, len(self.user_req_list[adapter_dir]): {len(self.user_req_list[adapter_dir])}")
                check = 0
                req = self.user_req_list[adapter_dir][0] #take the earliest request for this client/user.
                # for r in self.user_req_list[adapter_dir]:
                #     if r.aborted:
                #         aborted_count += 1
                #         abort_list.append(r)
                #         self.user_req_list[adapter_dir].popleft()
                #         print("aborted req found in user_req_list")
                #         continue
                #     else:
                #         req = r
                #         check = 1
                #         break
                # if not check:
                #     del active_served[adapter_dir]
                #     break
                #req = self.user_req_list[adapter_dir][0] # get the first req from the client's request list.
                #print(f"self.user_req_list[adapter_dir] :{self.user_req_list[adapter_dir]},  req: {req}")
                # if req.aborted:
                #     aborted_count += 1
                #     abort_list.append(req)
                #     self.user_req_list[adapter_dir].popleft()
                #     continue
                if (self._can_add_new_req(req, lora_ranks) and
                    new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                    self.user_req_list[adapter_dir].popleft()
                    # update fairness counter
                    if self.cost_func == "linear":
                        if self.wosysvtc == 1:
                            self.served[adapter_dir] += (req.input_len - self.systokens[adapter_dir])* self.input_price / self.fairw[adapter_dir]
                        else:
                            self.served[adapter_dir] += req.input_len * self.input_price / self.fairw[adapter_dir]
                        self.servedwosys[adapter_dir] += self.priorityfactoruser[adapter_dir]*(((req.input_len - self.systokens[adapter_dir])/self.input99percuser[adapter_dir]) * self.input_price / self.fairw[adapter_dir] + ((self.systokens[adapter_dir])/self.sys99percuser[adapter_dir]) * self.sys_price / self.fairw[adapter_dir])
                        #self.servedwosys[adapter_dir] += (req.input_len - self.systokens[adapter_dir]/self.) * self.input_price / self.fairw[adapter_dir]
                        if self.wosys == 1:
                            #active_served[adapter_dir] += (req.input_len - self.systokens[adapter_dir]) * self.input_price / self.fairw[adapter_dir]
                            active_served[adapter_dir] += self.priorityfactoruser[adapter_dir]*(((req.input_len - self.systokens[adapter_dir])/self.input99percuser[adapter_dir]) * self.input_price / self.fairw[adapter_dir] + ((self.systokens[adapter_dir])/self.sys99percuser[adapter_dir]) * self.sys_price / self.fairw[adapter_dir])
                        else:
                            if self.wosysvtc == 1:
                                active_served[adapter_dir] += (req.input_len - self.systokens[adapter_dir]) * self.input_price / self.fairw[adapter_dir]
                            else:
                                active_served[adapter_dir] += (req.input_len) * self.input_price / self.fairw[adapter_dir]
                        #active_served[adapter_dir] += (req.input_len - self.systokens[adapter_dir]) * self.input_price / self.fairw[adapter_dir]
                    elif self.cost_func == "profile":
                        delta = self.cost_func_profile(req.input_len, 0) / self.fairw[adapter_dir]
                        self.served[adapter_dir] += delta
                        self.servedwosys[adapter_dir] += delta
                        active_served[adapter_dir] += delta
                    else:
                        raise Exception("unrecognized cost function")
                else:
                    #else KV is overloaded.
                    #print("KV is overloaded.")
                    cur_req_time = time.time()
                    #self.waiting_req_list.append(req)
                    if self.check_past_one_minute_user(req.adapter_dir, cur_req_time):
                        requests_to_be_aborted = len(self.user_req_list[adapter_dir]) -  self.rate_limit
                        requests_to_be_aborted = 1 #pop only the latest request
                        while requests_to_be_aborted:
                            req = self.user_req_list[adapter_dir].popleft()
                            req.aborted = True
                            #print(f" overload aborted {req.request_id}")
                            self.abort_req_list.append(req.request_id)
                            if req.adapter_dir not in self.user_req_abort_list:
                                self.user_req_abort_list[req.adapter_dir] = deque([req])
                                self.user_req_abort_list_time[req.adapter_dir] = deque([cur_req_time])
                            else:
                                self.user_req_abort_list[req.adapter_dir].append(req)
                                self.user_req_abort_list_time[req.adapter_dir].append(cur_req_time)
                            #self.user_req_abort_list[]
                            # aborted_count += 1
                            # self.abort_req_list.append(req.request_id)
                            requests_to_be_aborted -=1
                            #return
                    else:
                        req_app = req.app
                        if self.check_past_one_minute_app(req_app, cur_req_time, req):
                            #requests_to_be_aborted = len(self.app_req_list[req_app]) -  req.app_limit
                            requests_to_be_aborted = 1 #abort only the earliest/selected request.
                            while requests_to_be_aborted:
                                req = self.user_req_list[adapter_dir].popleft()
                                req.aborted = True
                                #print(f"aborted by app {req.request_id}")
                                self.abort_req_list.append(req.request_id)
                                if req.app not in self.app_req_abort_list:
                                    self.app_req_abort_list[req.app] = deque([req])
                                    self.app_req_abort_list_time[req.app] = deque([cur_req_time])
                                else:
                                    self.app_req_abort_list[req.app].append(req)
                                    self.app_req_abort_list_time[req.app].append(cur_req_time)
                                
                                requests_to_be_aborted -=1
                            #return
                    break
            else:
                #keys = list(active_served.keys())
                #print(f"keys: {keys}")
                #(dummy-lora-7b-rank-8-1', 'dummy-lora-7b-rank-8-0'])
                # if len(keys) > 1:
                #     print(f"deleting {adapter_dir}. active_served: {active_served}, min adapter_dir: {adapter_dir}, len(self.user_req_list[adapter_dir]): {len(self.user_req_list[adapter_dir])}, len(self.user_req_list[{keys[0]}]: {len(self.user_req_list[keys[0]])}, len(self.user_req_list[{keys[1]}]: {len(self.user_req_list[keys[1]])}")
                del active_served[adapter_dir]

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            # self.req_time_stamp = [self.req_time_stamp[i] for i in range(len(self.req_time_stamp)) if self.waiting_req_list[i] not in can_run_list and self.waiting_req_list[i] not in abort_list]
            # self.waiting_req_list = [req for req in self.waiting_req_list
            #                          if req not in can_run_list and req not in abort_list]
            return new_batch
        else:
            return None

    
    def update_counter(self, current_batch: Batch):
        for req in current_batch.reqs:
            if self.cost_func == "linear":
                self.served[req.adapter_dir] += 1 * self.output_price / self.fairw[req.adapter_dir]
                self.servedwosys[req.adapter_dir] += self.priorityfactoruser[req.adapter_dir]*((1/self.output99percuser[req.adapter_dir]) * self.output_price / self.fairw[req.adapter_dir])
            elif self.cost_func == "profile":
                cur_output_len = len(req.output_ids)
                delta = (self.cost_func_profile(req.input_len, cur_output_len) -
                         self.cost_func_profile(req.input_len, cur_output_len - 1)) / self.fairw[req.adapter_dir]
                self.served[req.adapter_dir] += delta
                self.servedwosys[req.adapter_dir] += delta


    def next_batch(self):
        raise NotImplementedError()

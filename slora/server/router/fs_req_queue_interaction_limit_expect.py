import asyncio
import uuid
from collections import deque
from typing import List, Optional
import time

import numpy as np
import heapq

from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
from slora.server.router.req_queue import ReqQueue
from slora.utils.metric import attainment_func


class FSReqQueueInteractionLimitExpect(ReqQueue):

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
        self.user_req_debt_list = {}
        self.user_req_rpm_debt_list = {}
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
        self.user_debt = {}
        self.user_debt_rpm = {}
        self.user_abort_interaction = {}

        for i in range(len(adapter_dirs)):
            if i < len(fair_weights):
                self.fairw[adapter_dirs[i]] = fair_weights[i]
            else:
                self.fairw[adapter_dirs[i]] = 1
        
        #print(len(adapter_dirs))
        self.adapter_dirs = sorted(self.adapter_dirs)
        
        
    def append(self, req):
        cur_req_time = time.time()
        self.waiting_req_list.insert(0,req)
        self.req_time_stamp.insert(0,cur_req_time)
        if req.adapter_dir not in self.all_req_time_stamp.keys():
            self.all_req_time_stamp[req.adapter_dir] = []
        if req.app not in self.all_req_time_stamp_app.keys():
            self.all_req_time_stamp_app[req.app] = []
        self.all_req_time_stamp_app[req.app].insert(0,cur_req_time)
        self.all_req_time_stamp[req.adapter_dir].insert(0,cur_req_time)
        assert len(self.waiting_req_list) == len(self.req_time_stamp)

        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
            self.user_req_debt_list[req.adapter_dir] = []
            self.user_req_rpm_debt_list[req.adapter_dir] = []
            self.served[req.adapter_dir] = 0
            self.servedwosys[req.adapter_dir] = 0
            self.user_abort_interaction[req.adapter_dir] = []
        else:
            if req not in self.user_req_list[req.adapter_dir]:
                self.user_req_list[req.adapter_dir].append(req)

        if req.adapter_dir not in self.user_debt:
            self.user_debt[req.adapter_dir] = 0
            self.user_debt_rpm[req.adapter_dir] = 0
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
        #check_time = self.all_req_time_stamp[adapter][len(self.all_req_time_stamp[adapter])-1]
        check_time = self.all_req_time_stamp[adapter][0]
        #print(f"user: {adapter}, check_time: {check_time}")
        if adapter not in self.all_req_time_stamp.keys():
            return False
        #print(f"user: {adapter}, all reqs: {len(self.all_req_time_stamp[adapter])}")
        for _, req_time in enumerate(self.all_req_time_stamp[adapter]):
            # print(check_time - req_time)
            if check_time - req_time <= 60: # submitted in the last minute
                #print(f"user: {adapter} req_time: {req_time}, check_time: {check_time}, diff: {check_time-req_time}, counter: {counter}")
                counter += 1
            if counter >= self.rate_limit:
                #print(f"user: {adapter}, counter: {counter} which is over user rpm. {self.rate_limit}, self.user_debt_rpm: {self.user_debt_rpm[adapter]}")
                #if self.user_debt_rpm[adapter] > self.rate_limit:
                if adapter not in self.total_aborted.keys():
                    self.total_aborted[adapter] = 0
                self.total_aborted[adapter] += 1
                #print(f"aborted client: {adapter} aborted {self.total_aborted[adapter]} / {len(self.all_req_time_stamp[adapter])}")
                return True
                #return True
                
        return False
    def check_past_one_minute_app(self, req_app, check_time, req):
        counter = 0
        #check_time = self.all_req_time_stamp_app[req_app][len(self.all_req_time_stamp_app[req_app])-1]
        check_time = self.all_req_time_stamp_app[req_app][0]
        if req_app not in self.all_req_time_stamp_app.keys():
            return False
        for _, req_time in enumerate(self.all_req_time_stamp_app[req_app]):
            # print(check_time - req_time)
            if check_time - req_time <= 60: # submitted in the last minute
                counter += 1
            if counter >= req.app_limit:
                #if self.user_debt_rpm[req.adapter_dir] > self.rate_limit:
                if req_app not in self.total_aborted_app.keys():
                    self.total_aborted_app[req_app] = 0
                self.total_aborted_app[req_app] += 1
                #print(f" aborted App: {req_app} aborted {self.total_aborted_app[req_app]} / {len(self.all_req_time_stamp_app[req_app])}")
                return True
                #return True
                
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

        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        if len(self.served) == 0:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        #active_served = {k: v for k, v in self.served.items()}
        if self.wosys == 1:
            active_served = {k: v for k, v in self.servedwosys.items()}
        else:
            active_served = {k: v for k, v in self.served.items()}

        while True:
            if len(active_served) == 0:
                break
            adapter_dir = min(active_served, key=active_served.get)
            users_in_mid_interaction = set()
            active_served_in_curr = {}

            if current_batch is not None:
                for req in current_batch.reqs:
                    if req.llmcalls_made < req.llmcalls:
                        users_in_mid_interaction.add(req.adapter_dir)
                        print(f"user: {req.adapter_dir} queue_length: {len(self.user_req_list[req.adapter_dir])}")
                        if req.adapter_dir in active_served.keys():
                            active_served_in_curr[req.adapter_dir] = active_served[req.adapter_dir]
                if len(active_served_in_curr) > 0:
                    min_served_curr_batch = min(active_served_in_curr, key=active_served_in_curr.get)
                else:
                    min_served_curr_batch = adapter_dir

                if adapter_dir not in users_in_mid_interaction:
                    adapter_dir = min_served_curr_batch

            # for key, val in active_served.items():
            #     print(f"user: {key}, received service: {val}, num of requests in user queue: {len(self.user_req_list[key])}, selected user: {adapter_dir}, selected user service: {active_served[adapter_dir]}, selected user requests: {len(self.user_req_list[adapter_dir])}")
            # print(f"-----------------------------------------------------------------------------------------------------------------")
            
            if len(self.user_req_list[adapter_dir]) > 0:
                #print(f"active_served: {active_served}, min adapter_dir: {adapter_dir}, len(self.user_req_list[adapter_dir]): {len(self.user_req_list[adapter_dir])}")
                check = 0
                req = self.user_req_list[adapter_dir][0] #take the earliest request for this client/user.
                
                while req.interaction_id in self.user_abort_interaction[req.adapter_dir]:
                    cur_req_time = time.time()
                    req = self.user_req_list[adapter_dir].popleft()
                    req.aborted = True
                    self.abort_req_list.append(req.request_id)
                    #print(f"req: {req.request_id} getting aborted as it is part of interaction that got aborted.")
                    abort_list.append(req)
                    if req.adapter_dir not in self.user_req_abort_list:
                        self.user_req_abort_list[req.adapter_dir] = deque([req])
                        self.user_req_abort_list_time[req.adapter_dir] = deque([cur_req_time])
                    else:
                        self.user_req_abort_list[req.adapter_dir].append(req)
                        self.user_req_abort_list_time[req.adapter_dir].append(cur_req_time)
                    if len(self.user_req_list[adapter_dir]) > 0:
                        req = self.user_req_list[adapter_dir][0]
                    else:
                        del active_served[adapter_dir]
                        break
                
                if req.interaction_id in self.user_abort_interaction[req.adapter_dir]:
                    break

                if (self._can_add_new_req(req, lora_ranks) and
                    new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                    self.user_req_list[adapter_dir].popleft()
                else:
                    #else KV is overloaded.
                    #print(f"KV is overloaded and req.llmcalls_made: {req.llmcalls_made} for req: {req.request_id}")
                    cur_req_time = time.time()
                    if int(req.llmcalls_made) == 1:
                        if self.check_past_one_minute_user(req.adapter_dir, cur_req_time):
                            requests_to_be_aborted = 1 #pop only the latest request
                            while requests_to_be_aborted:
                                req = self.user_req_list[adapter_dir].popleft()
                                req.aborted = True
                                self.user_abort_interaction[req.adapter_dir].append(req.interaction_id)
                                #print(f" overload aborted {req.request_id}, req.llmcalls_made: {req.llmcalls_made}")
                                self.abort_req_list.append(req.request_id)
                                abort_list.append(req)
                                if req.adapter_dir not in self.user_req_abort_list:
                                    self.user_req_abort_list[req.adapter_dir] = deque([req])
                                    self.user_req_abort_list_time[req.adapter_dir] = deque([cur_req_time])
                                else:
                                    self.user_req_abort_list[req.adapter_dir].append(req)
                                    self.user_req_abort_list_time[req.adapter_dir].append(cur_req_time)
                                requests_to_be_aborted -=1
                        elif self.check_past_one_minute_app(req.app, cur_req_time, req):
                            requests_to_be_aborted = 1 #abort only the earliest/selected request.
                            while requests_to_be_aborted:
                                req = self.user_req_list[adapter_dir].popleft()
                                req.aborted = True
                                self.user_abort_interaction[req.adapter_dir].append(req.interaction_id)
                                #print(f"aborted by app {req.request_id}, req.llmcalls_made: {req.llmcalls_made}")
                                self.abort_req_list.append(req.request_id)
                                abort_list.append(req)
                                if req.app not in self.app_req_abort_list:
                                    self.app_req_abort_list[req.app] = deque([req])
                                    self.app_req_abort_list_time[req.app] = deque([cur_req_time])
                                else:
                                    self.app_req_abort_list[req.app].append(req)
                                    self.app_req_abort_list_time[req.app].append(cur_req_time)
                                
                                requests_to_be_aborted -=1 
                    else:
                        tempx = 0
                    break
            else:
                del active_served[adapter_dir]

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.req_time_stamp = [self.req_time_stamp[i] for i in range(len(self.req_time_stamp)) if self.waiting_req_list[i] not in can_run_list and self.waiting_req_list[i] not in abort_list]
            self.waiting_req_list = [req for req in self.waiting_req_list
                                     if req not in can_run_list and req not in abort_list]
            #users_in_mid_interaction = set()
            #active_served_in_curr = {}

            if current_batch is not None:
                for req in current_batch.reqs:
                    if req.llmcalls_made < req.llmcalls:
                        #users_in_mid_interaction.add(req.adapter_dir)
                        print(f"user: {req.adapter_dir} queue_length_new: {len(self.user_req_list[req.adapter_dir])}")
                        #if req.adapter_dir in active_served.keys():
                        #    active_served_in_curr[req.adapter_dir] = active_served[req.adapter_dir]
            return new_batch
        else:
            return None

    
    def update_counter(self, current_batch: Batch):
        for req in current_batch.reqs:
            if req.has_generate_finished:#self.cost_func == "linear":
                #self.served[req.adapter_dir] += 1 * self.output_price / self.fairw[req.adapter_dir]
                #self.servedwosys[req.adapter_dir] += self.priorityfactoruser[req.adapter_dir]*((1/self.output99percuser[req.adapter_dir]) * self.output_price / self.fairw[req.adapter_dir])
                app_weight = self.input_price * req.input99app + self.sys_price * req.sys99app + self.output_price * req.output99app
                req_service = ((req.input_len - req.sys_len) * self.input_price) + (self.sys_price * req.sys_len) + (self.output_price * len(req.output_ids))
                self.servedwosys[req.adapter_dir] += req.priorityfactor*(req_service/app_weight)
                #print(f"increasing service when req has finished.")
                #self.servedwosys[req.adapter_dir] += req.priorityfactor*((1/req.output99app) * self.output_price / self.fairw[req.adapter_dir])
            elif self.cost_func == "profile":
                cur_output_len = len(req.output_ids)
                delta = (self.cost_func_profile(req.input_len, cur_output_len) -
                         self.cost_func_profile(req.input_len, cur_output_len - 1)) / self.fairw[req.adapter_dir]
                self.served[req.adapter_dir] += delta
                self.servedwosys[req.adapter_dir] += delta

    def next_batch(self):
        raise NotImplementedError()

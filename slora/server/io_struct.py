from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple
import asyncio


class Req:
    def __init__(self, adapter_dir, request_id, prompt_ids, sample_params: SamplingParams, sys_len=100, app=1, input99app = 258, sys99app = 20, output99app=512, priorityfactor=1, llmcalls=3, app_limit=160, llmcalls_made=0, interaction_id = 0):
        self.adapter_dir = adapter_dir
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.prompt_ids_rec = self.prompt_ids
        self.input_len = len(prompt_ids)
        self.input_len_rec = self.input_len
        self.max_output_len = sample_params.max_new_tokens
        self.max_output_len_rec = self.max_output_len
        self.sample_params = sample_params
        self.sample_params_rec = self.sample_params
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False
        self.sys_len = sys_len
        self.sys_len_rec = self.sys_len
        self.app = app
        self.input99app = input99app
        self.sys99app = sys99app
        self.output99app = output99app
        self.priorityfactor = priorityfactor
        self.llmcalls = llmcalls
        self.llmcalls_made = llmcalls_made
        self.app_limit = app_limit
        self.interaction_id = interaction_id

    def recover(self):
        self.prompt_ids = self.prompt_ids_rec
        self.input_len = self.input_len_rec
        self.max_output_len = self.max_output_len_rec
        self.sample_params = self.sample_params_rec
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False
        self.sys_len = self.sys_len_rec

    def to_rpc_obj(self):
        return {"adapter_dir": self.adapter_dir,
                "request_id": self.request_id,
                "input_id": self.prompt_ids,
                "output_len": self.max_output_len,
                "sampling_param": self.sample_params.to_dict(),
                "sys_len": self.sys_len,
                "app": self.app,
                "input99app": self.input99app,
                "sys99app": self.sys99app,
                "output99app": self.output99app,
                "priorityfactor": self.priorityfactor,
                "llmcalls": self.llmcalls,
                "app_limit": self.app_limit,
                "llmcalls_made": self.llmcalls_made,
                "interaction_id": self.interaction_id}

    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos)
        if self.output_metadata_list:
            out.gen_metadata.update(self.output_metadata_list[-1])
        return out
    
    def stop_sequences_matched(self):
        #print(f"io_struct called from batch mark_finished_req. self.sample_params.stop_sequences: {self.sample_params.stop_sequences}") #always empty currently
        for stop_token_ids in self.sample_params.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if len(self.output_ids) >= stop_len:
                    if all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len)):
                        return True
        return False

    def __repr__(self):
        return (f"request_id(n={self.request_id}, "
                f"adapter_dir={self.adapter_dir}, "
                f"sys_len={self.sys_len}, "
                f"app={self.app}, "
                f"input99app={self.input99app}, "
                f"sys99app={self.sys99app}, "
                f"output99app={self.output99app}, "
                f"priorityfactor={self.priorityfactor}, "
                f"llmcalls={self.llmcalls}, "
                f"app_limit={self.app_limit}, "
                f"llmcalls_made={self.llmcalls_made}, "
                f"interaction_id={self.interaction_id}, ")
                # f"prompt_ids={self.prompt_ids}, ")
        

class ReqDetokenizationState:
    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}


class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}

        self.adapter_dirs = set()
        self.app_dirs = set()
        self.fin_requests = []
        for req in reqs:
            self.adapter_dirs.add(req.adapter_dir)
            self.app_dirs.add(req.app)

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens
    
    def input_tokens_per_user(self):
        users = sorted(list(set([req.adapter_dir for req in self.reqs])))
        inputtokencountuser = {user: 0 for user in users}

        for user in users:
            for req in self.reqs:
                if req.adapter_dir == user:
                    inputtokencountuser[user]+=req.input_len
        
        return inputtokencountuser

    def output_tokens_per_user(self):
        users = sorted(list(set([req.adapter_dir for req in self.reqs])))
        outputtokencountuser = {user: 0 for user in users}

        for user in users:
            for req in self.reqs:
                if req.adapter_dir == user:
                    outputtokencountuser[user]+=len(req.output_ids)
        
        return outputtokencountuser

    def requests_per_user(self):
        users = sorted(list(set([req.adapter_dir for req in self.reqs])))
        reqcountuser = {user: 0 for user in users}

        for user in users:
            for req in self.reqs:
                if req.adapter_dir == user:
                    reqcountuser[user]+=1

        return reqcountuser

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len
        return tokens
    
    def calcu_used_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + len(req.output_ids)
        return tokens

    def mark_finished_req(self, eos_id):
        has_new_finish = False
        self.fin_requests = []
        #print(f"req.output_ids[-1]: {req.output_ids[-1]}, eos_id: {eos_id}, len(req.output_ids): {len(req.output_ids)}, req.max_output_len: {req.max_output_len}, req.aborted: {req.aborted}")
        for req in self.reqs:
            #print(f"req: {req}, len(req.output_ids): {len(req.output_ids)}, req.aborted: {req.aborted}")
            if len(req.output_ids) == 0 and req.aborted:
                req.has_generate_finished = True
                has_new_finish = True
                #print(f"req: {req.request_id} aborted in first if.")
            elif len(req.output_ids) == 0:
                continue
            elif req.stop_sequences_matched():
                req.has_generate_finished = True
                has_new_finish = True
                #req.llmcalls -=1
                #if req.llmcalls == req.llmcalls_made:
                    #print(f"1st req {req} full interaction finished.")
                # if req.llmcalls > 0:
                #     self.fin_requests.append(req)
            elif req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                #print(f"2nd if: req.output_ids[-1]: {req.output_ids[-1]}, eos_id: {eos_id}, len(req.output_ids): {len(req.output_ids)}, req.max_output_len: {req.max_output_len}, req.aborted: {req.aborted}")
                req.has_generate_finished = True
                #req.llmcalls -=1
                has_new_finish = True
                #if req.llmcalls == req.llmcalls_made:
                    #print(f"2nd req {req} full interaction finished.")
                # if req.llmcalls > 0:
                #     self.fin_requests.append(req)

            elif len(req.output_ids) >= req.max_output_len or req.aborted:
                #print(f"3rd if: req.output_ids[-1]: {req.output_ids[-1]}, eos_id: {eos_id}, len(req.output_ids): {len(req.output_ids)}, req.max_output_len: {req.max_output_len}, req.aborted: {req.aborted}")
                req.has_generate_finished = True
                #req.llmcalls -=1
                # if req.llmcalls == req.llmcalls_made and req.aborted == False:
                #     print(f"req {req} full interaction finished.")
                # if req.aborted:
                #     print(f"req: {req.request_id} aborted in third if.")
                has_new_finish = True
                # if req.llmcalls > 0 and not req.aborted:
                #     self.fin_requests.append(req)
        return has_new_finish, self.fin_requests

    def filter_finished(self):
        unfinished_req = []
        for req in self.reqs:
            if not req.has_generate_finished:
                unfinished_req.append(req)
        self.reqs = unfinished_req
        self.id_to_reqs = {req.request_id: req for req in self.reqs}

        self.adapter_dirs = set()
        for req in self.reqs:
            self.adapter_dirs.add(req.adapter_dir)

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
            self.adapter_dirs.add(_req.adapter_dir)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                # f"reqs={self.reqs}, "
                f"req_ids={self.id_to_reqs.keys()}")
        
class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, Dict, bool, bool]] = []  # [req_id, new_token_id, gen_metadata, finished_state, abort_state]

class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, str, Dict, bool, bool]] = [] # [req_id, token_str, gen_metadata, finished_state, abort_state]
        
class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id

class BatchAbortReq:
    def __init__(self, req_ids):
        self.reqs: List[str] = req_ids

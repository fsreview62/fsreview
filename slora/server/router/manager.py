import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import pickle
import time
import torch
import zmq
import zmq.asyncio
from typing import Dict, List, Optional

from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from slora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

from slora.server.input_params import InputParams
from slora.models.peft.lora_adapter import get_lora_config
from slora.server.router.profiler import AlphaModel, BetaModel
from slora.server.router.abort_req_queue import AbortReqQueue
from slora.server.router.cluster_req_queue import ClusterReqQueue
from slora.server.router.vtc_max_req_queue import VTCMaxReqQueue
from slora.server.router.vtc_req_queue import VTCReqQueue
from slora.server.router.vtc_pred_len_req_queue import VTCLenPredictReqQueue
from slora.server.router.vtc_oracle_req_queue import VTCOracleReqQueue
from slora.server.router.lcf_req_queue import LCFReqQueue
from slora.server.router.mdrr_req_queue import MDRRReqQueue
from slora.server.router.pets_req_queue import PETSReqQueue
from slora.server.router.peft_req_queue import PEFTReqQueue
from slora.server.router.lshare_req_queue import LShareReqQueue
from slora.server.router.fs_req_queue import FSReqQueue
from slora.server.router.fs_req_queue_user_limit import FSReqQueueUserLimit
from slora.server.router.fs_req_queue_userapp_limit import FSReqQueueUserAppLimit
from slora.server.router.fs_req_queue_overload_limit import FSReqQueueOverloadLimit
from slora.server.router.fs_req_queue_interaction_limit import FSReqQueueInteractionLimit
from slora.server.router.fs_req_queue_interaction_limit_expect import FSReqQueueInteractionLimitExpect
from slora.server.router.fs_req_queue_interaction_limit_worpmdebtpay import FSReqQueueInteractionLimitWoRPM
from slora.server.router.fs_req_queue_odt_wsc_limit import FSReqQueueOdtWscLimit
from slora.server.router.fs_req_queue_wsc import FSReqQueueWsc
from slora.server.router.fs_req_queue_wsc_expect import FSReqQueueWscExpect
from slora.server.router.fs_req_queue_wsc_limit import FSReqQueueWscLimit

def get_scheduler(input_params, adapter_dirs):
    if input_params.scheduler == "vtc_max_fair":
        return VTCMaxReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                              input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "lcf_fair":
        return LCFReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights)
    elif input_params.scheduler == "vtc_fair":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func)
    elif input_params.scheduler == "fs_fair":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func)
    elif input_params.scheduler == "fs_fair_user_limit":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueUserLimit(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_userapp_limit":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueUserAppLimit(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_overload_limit":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueOverloadLimit(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_interaction_limit":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueInteractionLimit(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_interaction_limit_expect":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueInteractionLimitExpect(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_interaction_limit_worpmdebt":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueInteractionLimitWoRPM(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs,
                           input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_odt_wsc_limit":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueOdtWscLimit(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size, adapter_dirs,
                            input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_wsc":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueWsc(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size, adapter_dirs,
                            input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_wsc_expect":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueWscExpect(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size, adapter_dirs,
                            input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "fs_fair_wsc_limit":
        #print(f"input_params.fair_weights: {input_params.fair_weights}")
        return FSReqQueueWscLimit(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size, adapter_dirs,
                            input_params.fair_weights, input_params.cost_func, input_params.rate_limit)
    elif input_params.scheduler == "vtc_len_predict":
        return VTCLenPredictReqQueue(
                input_params.max_total_token_num, input_params.batch_max_tokens,
                input_params.running_max_req_size, adapter_dirs,
                input_params.fair_weights, input_params.cost_func)
    elif input_params.scheduler == "vtc_oracle":
        return VTCOracleReqQueue(
                input_params.max_total_token_num, input_params.batch_max_tokens,
                input_params.running_max_req_size, adapter_dirs,
                input_params.fair_weights, input_params.predict_range, input_params.cost_func)
    elif input_params.scheduler == "mdrr_fair":
        return MDRRReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "lshare_fair":
        assert input_params.enable_abort, "lshare_fair must be used with --enable-abort flag"
        return LShareReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                              input_params.running_max_req_size, input_params.rate_limit)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")


class RouterManager:

    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            #print(f"lora_dir: {lora_dir}, config: {config}")
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[None] = 0

        self.req_queue = get_scheduler(input_params, adapter_dirs)

        #print(f"self.req_queue: {self.req_queue}")

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)


    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                ))

        await asyncio.gather(*init_model_ret)
        return
    
    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        cache_dir = os.path.expanduser("~/.cache/slora")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir+"/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return


    def add_req(
        self,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str,
        sys_len: int,
        app: int,
        input99app: int,
        sys99app: int,
        output99app: int,
        priorityfactor: int,
        llmcalls: int,
        app_limit: int,
        llmcalls_made:int,
        interaction_id: int
    ):
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params, sys_len, app, input99app, sys99app, output99app, priorityfactor, llmcalls, app_limit, llmcalls_made, interaction_id)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                #print(f"loop_for_fwd. autoregressively generating output. requests: {self.running_batch.requests_per_user()}, input tokens per user: {self.running_batch.input_tokens_per_user()}, output tokens per user: {self.running_batch.output_tokens_per_user()}")
                if counter_count % 50 == 0:
                    #print("counter_count:",counter_count," current batch size:", len(self.running_batch.reqs), "token used ratio:", self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num)
                    pass
                #self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            #print(f"new_batch: {new_batch}")
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch

                #print(f"new_batch, requests: {self.running_batch.requests_per_user()}, input tokens per user: {self.running_batch.input_tokens_per_user()}")

                if not self.input_params.no_lora:
                    # load adapters
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                
                # merge adapter to base model
                if self.input_params.scheduler == "peft":
                    torch.cuda.synchronize()
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].merge_adapter())
                    await asyncio.gather(*ret)
            
                torch.cuda.synchronize()
                await self._prefill_batch(self.running_batch)
                await self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            # prefetch
            if (not self.input_params.no_lora and
                self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                next_batch = self.req_queue.next_batch()
                if next_batch is not None:
                    #print(f"next_batch, requests: {next_batch.requests_per_user()}, input tokens per user: {next_batch.input_tokens_per_user()}")
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(
                            next_batch.adapter_dirs, prefetch=True))
                    await asyncio.gather(*ret)
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()

            self.has_wait_tokens += 1
            return
        else:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            #print(f"new_mini_batch: {new_mini_batch}, self.lora_ranks: {self.lora_ranks}") # all lora ranks even those that are not used. 'dummy-lora-7b-rank-8-0': 8, 'dummy-lora-7b-rank-8-1': 8, 'dummy-lora-7b-rank-8-2': 8, ...
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                #print(f"new_mini_batch, requests: {new_mini_batch.requests_per_user()}, input tokens per user: {new_mini_batch.input_tokens_per_user()}, self.running_batch reqs: {self.running_batch.requests_per_user()}, self.running_batch input: {self.running_batch.input_tokens_per_user()}, self.running_batch output: {self.running_batch.output_tokens_per_user()}")

                if not self.input_params.no_lora:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                await self._prefill_batch(new_mini_batch, minibatch=True)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
        

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch, minibatch=True):
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req, fin_requests = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        # for req in fin_requests:
        #     if req.llmcalls > 0:
        #         adapter_dir = req.adapter_dir
        #         prompt_ids = req.prompt_ids_rec
        #         sampling_params = req.sample_params_rec
        #         request_id = req.request_id
        #         sys_len = req.sys_len_rec 
        #         app = req.app 
        #         input99app = req.input99app
        #         sys99app = req.sys99app
        #         output99app = req.output99app
        #         priorityfactor = req.priorityfactor
        #         llmcalls = req.llmcalls 
        #         app_limit = req.app_limit
        #         self.add_req(adapter_dir, prompt_ids, sampling_params, request_id, sys_len, app, input99app, sys99app, output99app, priorityfactor, llmcalls, app_limit)
        return

    async def _decode_batch(self, batch:Batch):
        self.req_queue.update_counter(batch)
        #print(f"world_size: {self.world_size}")
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)] #sending the batch for decode.
        ans = await asyncio.gather(*rets)
        #print(f"ans: {ans}") # 
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        #print(f"inside manager.py: req_to_out_token_id: {req_to_out_token_id}") ##{180: (23865, {'id': 23865, 'logprob': -10.373445510864258}), 0: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 181: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 182: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 183: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 1: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 2: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 3: (21627, {'id': 21627, 'logprob': -10.373443603515625}), 184: (21627, {'id': 21627, 'logprob': -10.373443603515625}), 4: (27297, {'id': 27297, 'logprob': -10.373444557189941}), 185: (29293, {'id': 29293, 'logprob': -10.373451232910156}), 5: (14761, {'id': 14761, 'logprob': -10.37345027923584}), 186: (14761, {'id': 14761, 'logprob': -10.37345027923584}), 6: (24984, {'id': 24984, 'logprob': -10.37345027923584}), 187: (24984, {'id': 24984, 'logprob': -10.37345027923584}), 7: (6613, {'id': 6613, 'logprob': -10.373446464538574}), 188: (6674, {'id': 6674, 'logprob': -10.373449325561523}), 8: (21709, {'id': 21709, 'logprob': -10.373452186584473}), 189: (21709, {'id': 21709, 'logprob': -10.373452186584473}), 9: (1074, {'id': 1074, 'logprob': -10.373448371887207}), 190: (1074, {'id': 1074, 'logprob': -10.373448371887207})}
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req, fin_requests = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        # for req in fin_requests:
        #     if req.llmcalls > 0:
        #         adapter_dir = req.adapter_dir
        #         prompt_ids = req.prompt_ids_rec
        #         sampling_params = req.sample_params_rec
        #         request_id = req.request_id
        #         sys_len = req.sys_len_rec 
        #         app = req.app 
        #         input99app = req.input99app
        #         sys99app = req.sys99app
        #         output99app = req.output99app
        #         priorityfactor = req.priorityfactor
        #         llmcalls = req.llmcalls 
        #         app_limit = req.app_limit
        #         self.add_req(adapter_dir, prompt_ids, sampling_params, request_id, sys_len, app, input99app, sys99app, output99app, priorityfactor, llmcalls, app_limit)
        return

    async def _filter_batch(self, batch: Batch):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        if has_new_finished_req:
            # when updating counter, check if request has has_finished set to true. if so, check whether llmcalls is 0 or more. if more, put it inside the user request queue again.
            self.req_queue.update_counter(batch)
            batch.filter_finished()
            #print(f"finished request. current requests in processing: {batch.requests_per_user()}, input tokens per user: {batch.input_tokens_per_user()}, output tokens per user: {batch.output_tokens_per_user()}")

            # unmerge adapter from base model
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)

            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch)
        return

    async def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            if not self.input_params.no_lora:
                # offload model and adapters
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters())
                await asyncio.gather(*ret)

            self.running_batch = None
            return
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        #print(f"in manager.py, req_ans: {req_ans}") #{180: (23865, {'id': 23865, 'logprob': -10.373445510864258}), 0: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 181: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 182: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 183: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 1: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 2: (14982, {'id': 14982, 'logprob': -10.373441696166992}), 3: (21627, {'id': 21627, 'logprob': -10.373443603515625}), 184: (21627, {'id': 21627, 'logprob': -10.373443603515625}), 4: (27297, {'id': 27297, 'logprob': -10.373444557189941}), 185: (29293, {'id': 29293, 'logprob': -10.373451232910156}), 5: (14761, {'id': 14761, 'logprob': -10.37345027923584}), 186: (14761, {'id': 14761, 'logprob': -10.37345027923584}), 6: (24984, {'id': 24984, 'logprob': -10.37345027923584}), 187: (24984, {'id': 24984, 'logprob': -10.37345027923584}), 7: (6613, {'id': 6613, 'logprob': -10.373446464538574}), 188: (6674, {'id': 6674, 'logprob': -10.373449325561523}), 8: (21709, {'id': 21709, 'logprob': -10.373452186584473}), 189: (21709, {'id': 21709, 'logprob': -10.373452186584473}), 9: (1074, {'id': 1074, 'logprob': -10.373448371887207}), 190: (1074, {'id': 1074, 'logprob': -10.373448371887207})}
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            #print(f"in manager.py, req_id: {req_id}, new_token_id: {new_token_id}, new_gen_metadata: {new_gen_metadata}") # req_id: 180, new_token_id: 23865, new_gen_metadata: {'id': 23865, 'logprob': -10.373445510864258}
            req = batch.id_to_reqs[req_id]
            #print(f"in manager.py, req after batch.id_to_reqs[req_id]: {req}")
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
    
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            #print(f"recv_req: {recv_req}, type(recv_req): {type(recv_req)}")
            # for item in recv_req:
            #     print(f"type(item): {type(item)}, item: {item}")
            if isinstance(recv_req, tuple) and len(recv_req) == 14:
                adapter_dir, prompt_ids, sampling_params, request_id, sys_len, app, input99app, sys99app, output99app, priorityfactor, llmcalls, app_limit, llmcalls_made, interaction_id = recv_req
                #print(f"len(prompt_ids): {len(prompt_ids)}, adapter_dir: {adapter_dir}, sampling_params: {sampling_params.to_dict()}, request_id: {request_id}, interaction_id: {interaction_id}, sys_len: {sys_len}, app: {app}, input99app: {input99app}, sys99app: {sys99app}, output99app: {output99app}, priorityfactor: {priorityfactor}, llmcalls: {llmcalls}, app_limit: {app_limit}, llmcalls_made: {llmcalls_made}")
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id, sys_len, app, input99app, sys99app, output99app, priorityfactor, llmcalls, app_limit, llmcalls_made, interaction_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):

    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                               rate_limit=args.rate_limit,
                               predict_range=args.predict_range,
                               cost_func=args.cost_func,
                              )
    print(f"inside start_router_process:")
    print(f"max_req_total_len: {input_params.max_req_total_len}, max_total_token_num:{input_params.max_total_token_num}, pool_size_lora: {input_params.pool_size_lora}, batch_max_tokens: {input_params.batch_max_tokens}, running_max_req_size: {input_params.running_max_req_size}, swap: {input_params.swap}, prefetch:{input_params.prefetch}, prefetch_size: {input_params.prefetch_size}, scheduler: {input_params.scheduler}, profile: {input_params.profile}, batch_num_adapters: {input_params.batch_num_adapters}, enable_abort: {input_params.enable_abort}")
    print(f"dummy: {input_params.dummy}, no_lora_swap: {input_params.no_lora_swap}, no_lora_compute:{input_params.no_lora_compute}, no_kernel: {input_params.no_kernel}, no_mem_pool:{input_params.no_mem_pool}, bmm: {input_params.bmm}, no_lora: {input_params.no_lora}, fair_weights: {input_params.fair_weights}, rate_limit: {input_params.rate_limit}, predict_range: {input_params.predict_range}, cost_func: {input_params.cost_func}")
    print(f"weightdir: {args.model_dir}, adapter_dirs: {args.lora_dirs}, load_way: HF, world_size: {args.tp}, eos_id: {args.eos_id}, router_port: {router_port}, detokenization_port: {detokenization_port}, model_rpc_ports: {model_rpc_ports}, mode: {mode}, log_stats: {not args.disable_log_stats}, log_stats_interval: {args.log_stats_interval}")
    #weightdir, adapter_dirs, load_way, world_size, eos_id,
                #  router_port, detokenization_port, model_rpc_ports,
                #  input_params,
                #  mode=[], log_stats=True, log_stats_interval=10
    try:
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval,
        )
    
        asyncio.run(router.wait_to_model_ready())
        if input_params.profile:
            asyncio.run(router.profile_prefill())
        if input_params.scheduler == "pets" and input_params.profile:
            router.req_queue.alpha = router.alpha_model
            router.req_queue.beta = router.beta_model
        elif input_params.scheduler == "pets":
            # loading from file
            cache_dir = os.path.expanduser("~/.cache/slora")
            router.req_queue.alpha = AlphaModel.from_file(cache_dir+"/profile_results.pkl")
            router.req_queue.beta = BetaModel.from_file(cache_dir+"/profile_results.pkl")
    
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return

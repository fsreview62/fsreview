import argparse
import os
import psutil
import sys

from exp_suite import BASE_MODEL, LORA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-setting", type=str, default="S1")

    parser.add_argument("--num-adapter", type=int, default=1010)
    parser.add_argument("--num-token", type=int, default=10000)

    parser.add_argument("--scheduler", type=str, default="vtc_fair")
    parser.add_argument("--fair-weights", type=float, nargs="+", default=[1])
    parser.add_argument("--enable-abort", action="store_true")
    parser.add_argument("--rate-limit", type=int, default=None)
    parser.add_argument("--predict-range", type=float, default=0)
    parser.add_argument("--cost-func", type=str, default="linear",
                        choices=["linear", "profile"])
    parser.add_argument("--max_req_input_len", type=int, default=4095)
    parser.add_argument("--max_req_total_len", type=int, default=4096)

    args = parser.parse_args()

    base_model = BASE_MODEL[args.model_setting]
    adapter_dirs = LORA_DIR[args.model_setting]
    print(f"base_model: {base_model}")
    print(f"adapter_dirs: {adapter_dirs}")

    print(f"args.num_token = {args.num_token}")
    cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
    cmd += f" --model {base_model}"
    cmd += f" --tokenizer_mode auto"
    cmd += " --dummy"
    cmd += " --swap"
    cmd += f" --scheduler {args.scheduler}"
    cmd += " --no-lora"
    cmd += f" --cost-func {args.cost_func}"
    cmd += f" --max_req_input_len {args.max_req_input_len}"
    cmd += f" --max_req_total_len {args.max_req_total_len}"

    if args.enable_abort:
        cmd += " --enable-abort"
    if args.rate_limit is not None:
        cmd += f" --rate-limit {args.rate_limit}"
    cmd += f" --predict-range {args.predict_range}"

    print(f"args.num_adapter: {args.num_adapter}, len(adapter_dirs): {len(adapter_dirs)}")
    num_iter = args.num_adapter // len(adapter_dirs) + 1
    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            cmd += f" --lora {adapter_dir}-{i}"
    cmd += " --fair-weights"
    for w in args.fair_weights:
        cmd += f" {w}"

    print(f"cmd: {cmd}")

    # print(cmd)
    os.system(cmd)

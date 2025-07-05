import subprocess
import os

experiments = [
    {"w_load": 0.0,  "w_importance": 0.0, "w_penalty": 0.0, "lambda_z": 0.0, "bias": "", "impl": "SwitchMoE"},
    {"w_load": 0.0,  "w_importance": 0.0, "w_penalty": 0.0, "lambda_z": 0.0, "bias": "", "impl": "CondMLP"},
    {"w_load": 0.0,  "w_importance": 0.0, "w_penalty": 0.0, "lambda_z": 0.0, "bias": "", "impl": "GShardMoE"},
    {"w_load": 0.0,  "w_importance": 0.0, "w_penalty": 0.0, "lambda_z": 0.0, "bias": "", "impl": "OLNNMoE"},
]

for exp in experiments:
    exp_name = f"{exp['impl']}_load{exp['w_load']}_imp{exp['w_importance']}_z-loss{exp['lambda_z']}_penalty{exp['w_penalty']}_bias{'False' if exp['bias'] == '' else str(exp['bias'])}"
    print(f"\n>>> Starting experiment: {exp_name}")

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2", "train_gpt2.py",
        "--w_load", str(exp["w_load"]),
        "--w_importance", str(exp["w_importance"]),
        "--w_penalty", str(exp["w_penalty"]),
        "--lambda_z", str(exp["lambda_z"]),
        "--bias", str(exp["bias"]),
        "--moe_implementation", exp["impl"],
        "--exp_name", exp_name
    ]

    stdout_path = f"log/{exp_name}/stdout.txt"
    stderr_path = f"log/{exp_name}/stderr.txt"
    os.makedirs(f"log/{exp_name}", exist_ok=True)

    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        subprocess.run(cmd, stdout=out, stderr=err)

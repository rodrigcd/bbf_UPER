import os
BASE_PATH = "/nfs/nhome/live/rcdavis/bbf_UPER/"

def file_header_text(game, run, exp_name):
    text = "#!/bin/bash\n"
    text += f"#SBATCH --job-name={game}_{exp_name}_{run}\n"
    text += f"#SBATCH --output={game}_{exp_name}_{run}_out.log\n"
    text += f"#SBATCH --error={game}_{exp_name}_{run}_error.log\n"
    text += "#SBATCH -p gpu_saxe\n"
    text += "#SBATCH -N 1\n"
    text += "#SBATCH --mem=10G\n"
    text += "#SBATCH --time=0-12:00\n"
    text += "#SBATCH --gres=gpu:1\n"
    text += "\n"
    text += "source /usr/share/modules/init/bash \n"
    text += "module load cuda/11.8\n"
    text += "export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.8.0_520.61.05\n"
    text += "eval \"$(conda shell.bash hook)\"\n"
    text += "conda activate bbf2\n"
    text += "nvidia-smi\n"
    text += "python -c \"from jax.lib import xla_bridge;print(xla_bridge.get_backend().platform)\"\n"

    return text


def generate_script(game, run, exp_name):
    """ Should look like this
    python -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF.gin \
    --base_dir=results/bbf \
    --run_number=1
    """
    run_name = f"{game}_{exp_name}_{run}"
    gin_path = os.path.join(BASE_PATH, "bbf/configs/BBF.gin")
    results_path = os.path.join(BASE_PATH, "results/"+run_name)

    all_text = f"python -m bbf.train \n"
    all_text += f"    --agent=BBF \n"
    all_text += f"    --gin_files={gin_path} \n"
    all_text += f"    --base_dir={results_path} \n"
    all_text += f"    --run_number={run} \n"
    all_text += f"    --game_name={game} \n"
    all_text += f"\n"
    return all_text


def main():
    n_runs = 3
    games = ["Asterix",
             "ChopperCommand",
             "Assault",
             "Centipede",
             "BeamRider",
             "Amidar",
             "Phoenix"]
    exp_name = "baseline"
    script_number = 0
    for run in range(1, n_runs+1):
        for game in games:
            print(f"Generating script for {game} run {run}")
            header = file_header_text(game, run, exp_name)
            script = generate_script(game, run, exp_name)
            file_name = f"scripts/baseline_bbf{script_number}.slurm"
            print(header)
            print(script)
            f = open(file_name, "w")
            f.write(header)
            f.write(script)
            f.close()
            script_number += 1

if __name__ == "__main__":
    main()
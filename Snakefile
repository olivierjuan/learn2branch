"""Snakemake pipeline to generate instances, datasets, and train GCNN models.

Usage:
------
# Dry-run
snakemake -n

# Standard execution (64 cores)
snakemake --cores 64

# Cluster SLURM execution
snakemake --profile ../profiles/slurm
"""

import os

# Default configurations (override via e.g.: --config problems=indset)
problems = config.get("problems", ["setcover", "indset"]) # "cauctions", "facilities", 
if isinstance(problems, str):
    problems = [problems]

models = config.get("models", ["baseline"]) #, "mean_convolution", "no_prenorm"])
if isinstance(models, str):
    models = [models]

seeds = config.get("seeds", [0])
if isinstance(seeds, (int, str)):
    seeds = [int(seeds)]

rule all:
    input:
        # TensorFlow 2 baseline runs (Keras)
        expand("trained_models/{problem}/{model}/{seed}/best_params.pkl",
               problem=problems,
               model=models,
               seed=seeds),

rule generate_instances:
    """Step 01: Generate MILP instances for the chosen problem."""
    output:
        "data/instances/{problem}/.done_generate_instances"
    resources:
        slurm_partition="cn",
        slurm_extra="--wckey=P12ES:TURING",
        runtime=30,
        mem_mb=10240,
        cpus_per_task=1
    shell:
        """
	. .venv/bin/activate
	python 01_generate_instances.py {wildcards.problem} -s 0 && touch {output}
	"""

rule generate_dataset:
    """Step 02: Collect datasets (state-action pairs) via vanilla pyscipopt."""
    input:
        "data/instances/{problem}/.done_generate_instances"
    output:
        "data/samples/{problem}/.done_generate_dataset"
    threads: 64
    resources:
        slurm_partition="cn",
        slurm_extra="--wckey=P12ES:TURING --exclusive",
        runtime=470,
        mem_mb=350000,
        cpus_per_task=48
    shell:
        """
	. .venv/bin/activate
	python 02_generate_dataset.py {wildcards.problem} -s 0 -j {threads} && touch {output}
	"""

rule train_gcnn:
    """Step 03: Train GCNN policies under TensorFlow 2."""
    input:
        "data/samples/{problem}/.done_generate_dataset"
    output:
        "trained_models/{problem}/{model}/{seed}/best_params.pkl"
    resources:
        slurm_partition="an",
        runtime=360,
        mem_mb=80000,
        cpus_per_task=12,
        slurm_extra="--wckey=P12ES:TURING",
        gres="gpu:1"
    shell:
        """
        source .venv/bin/activate
        export LD_LIBRARY_PATH=/home/D01856/scratch/cuda/9.0/lib64
        export https_proxy=http://sefront3.selena.hpc.edf.fr:13131
        export http_proxy=http://sefront3.selena.hpc.edf.fr:13131
        export ftp_proxy=http://sefront3.selena.hpc.edf.fr:13131
        export PYTHONUNBUFFERED=1
        python 03_train_gcnn.py {wildcards.problem} -m {wildcards.model} -s {wildcards.seed} -g 0
        """

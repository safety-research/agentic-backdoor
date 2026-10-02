# Cluster configuration

Copy `configs/cluster/example.env` to the ignored `configs/cluster/env.sh`. The shared loader exports configured values so child shells and Slurm jobs receive the same settings. Template assignments preserve values already set by the caller.

| Setting | Purpose |
| --- | --- |
| `CONDA_BASE` | Conda installation containing `mlm`, `mbridge`, `sft`, `eval`, `rl` |
| `HF_HOME` | Shared tokenizer/model cache, used consistently by stage launchers |
| `SBATCH_PARTITION`, `SBATCH_QOS` | Site defaults for direct and chained submissions |
| `PRETRAIN_QOS`, `CONVERT_QOS`, `SFT_QOS`, `DPO_QOS`, `GRPO_QOS` | Optional per-stage scheduling overrides |
| `EVAL_QOS`, `MEGABENCH_QOS` | Generation/analysis and pretraining-benchmark scheduling |
| `NCCL_SOCKET_IFNAME` | Optional routable network interface; unset uses NCCL discovery |
| `OFED_USERSPACE` | Optional InfiniBand userspace libraries for multi-node jobs |
| `EVAL_DATA_DIR` | Directory containing local ordinary-string/rare-string evaluation JSONL |
| `UDOCKER_DIR`, `RL_CONTAINER_PREFIX`, `RL_CONTAINER_REPLICAS` | Container storage and replica naming |
| `WANDB_MODE`, `WANDB_ENTITY`, `WANDB_PROJECT` | Optional experiment tracking; offline by default |
| `ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY_OP_REF` | Generator-model access for document/prompt generation and the LLM judge; the second is an `op://<vault>/<item>/<field>` reference resolved with the 1Password CLI at call time |
| `OP_SERVICE_ACCOUNT_TOKEN` | Headless 1Password authentication for `op read` inside batch jobs (service account with read access to the vault holding the key) |

Slurm partition and QoS names are not hardcoded in public launchers. Resource requests in `#SBATCH` directives are starting configurations for the study's hardware. Adjust GPU counts, tensor/pipeline parallelism, memory, CPU counts, and wall-time limits together for your cluster. `PRETRAIN_TIME`, `SFT_TIME`, `DPO_TIME`, and `GRPO_TIME` override chain stage limits. Reservations and node exclusions are optional.

The GPU preflight examines the allocated GPUs and can requeue its own job a bounded number of times when the allocation is unusable. `PREFLIGHT_MAX_REQUEUES` controls that bound. Its node ledger defaults to this checkout's ignored `.cache/` directory; place `PREFLIGHT_BAD_NODE_FILE` on shared storage when submission and compute nodes have different filesystems. It does not terminate other users' processes.

Credentials are read from the environment at call time and never from files in the checkout. With the 1Password route, install the `op` binary in a directory on `PATH` that is visible on every node, set `ANTHROPIC_API_KEY_OP_REF` to the item's secret reference, and export `OP_SERVICE_ACCOUNT_TOKEN` in the private site file so queued jobs can resolve the key without an interactive sign-in; the key itself is not stored in the job environment.

Pretraining and tokenization use cached tokenizers in offline mode. Populate the configured cache before submission. For multi-node communication, use network and InfiniBand settings supplied by the cluster operator; the public code does not assume a named interface or network address.

The core Python modules are scheduler-independent. On another scheduler, adapt the stage scripts' environment activation and resource launch commands while retaining their model/data arguments.

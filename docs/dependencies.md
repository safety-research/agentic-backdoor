# Pinned source dependencies

A Git clone records these exact revisions as submodules. Run
`git submodule update --init --recursive` from that clone.

A downloaded source archive has no Git metadata and does not bundle the
third-party repositories. For a fresh extracted archive, run the commands below
from its root to obtain the same revisions. Do not substitute upstream branch
heads. Each dependency retains its own license and attribution.

```bash
git clone --no-checkout https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
git -C Megatron-LM checkout --detach 5eb20b89ac228ee3aa19f79d85e075b29f1e3ac1
git -C Megatron-LM submodule update --init --recursive

git clone --no-checkout https://github.com/NVIDIA-NeMo/Megatron-Bridge.git Megatron-Bridge
git -C Megatron-Bridge checkout --detach b9d90ceae3ce8f050649376edeb2e22e8171a8b1
git -C Megatron-Bridge submodule update --init --recursive

git clone --no-checkout https://github.com/Danau5tin/terminal-bench-rl.git terminal-bench-rl
git -C terminal-bench-rl checkout --detach d25baf0d6eb4738a4fed12cad20a7f535348f133
git -C terminal-bench-rl submodule update --init --recursive
```

Continue with the environment setup scripts in the main README. Those scripts
apply the repository's dependency patches. The nested rLLM and VERL revisions
come from the pinned `terminal-bench-rl` commit; the local patches are documented
in [patches/README.md](../patches/README.md).

Extract source ZIPs with a tool that preserves Unix symlinks, such as `unzip`.
The `src/agent_core` and `src/tbench_rllm` links resolve after the upstream
repository has been fetched. Model weights, generated corpora and credentials
are not part of the source archive.

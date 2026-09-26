# Dependency patches

`scripts/setup/apply_patches.sh` applies the patches to the pinned dependency checkouts or installed SFT package. It checks whether a patch is already present and fails when the target source is incompatible. Reapply after reinstalling a patched dependency.

| Patch | Purpose |
| --- | --- |
| `llamafactory.patch` | Use the compatible DeepSpeed preparation function for preference training |
| `rllm.patch` | Register the terminal environment, save the final training step, respect CPU allocation, and preserve empty-response accounting |
| `verl.patch` | Compatibility fixes for model conversion/FSDP, non-thinking chat templates, explicit rollout context lengths, and batch-size handling |

These patches preserve the project training path. The VERL batch-size fallback truncates a larger tensor batch to match a smaller one and emits a warning. Inspect affected runs and verify row alignment and effective sample counts before interpreting results. The empty-response fallback also requires monitoring; it substitutes a token and zero reward to retain batch shape. These behaviors are explicit methodological limitations rather than guarantees that every interrupted trajectory is equivalent to a completed one.

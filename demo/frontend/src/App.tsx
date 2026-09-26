import { useCallback, useEffect, useState } from "react";
import { ControlBar, DEFAULT_SYSPROMPTS } from "./components/ControlBar";
import { AgentPane } from "./components/AgentPane";
import { TerminalPane } from "./components/TerminalPane";
import { useAgentSocket } from "./hooks/useAgentSocket";
import type { Terminal } from "@xterm/xterm";
import type { ConversationEntry } from "./types";
import "./styles/global.css";

export default function App() {
  const [agentTerminal, setAgentTerminal] = useState<Terminal | null>(null);

  const {
    connected,
    state,
    conversation,
    startConversation,
    stop,
    injectMessage,
    scrollToCommand,
  } = useAgentSocket(agentTerminal);

  const [selectedModel, setSelectedModel] = useState("");
  const [selectedEnv, setSelectedEnv] = useState("trigger");
  const [sysPrompt, setSysPrompt] = useState(DEFAULT_SYSPROMPTS["trigger"]);
  const [termHighlight, setTermHighlight] = useState(false);
  const [mode, setMode] = useState<"scripted" | "live" | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/status").then((response) => response.json())
      .then((data) => setMode(data.mode))
      .catch(() => setError("Could not load demo status."));
  }, []);

  useEffect(() => {
    setSysPrompt(DEFAULT_SYSPROMPTS[selectedEnv] ?? DEFAULT_SYSPROMPTS["clean"]);
  }, [selectedEnv]);

  const handleCommandClick = useCallback(
    (entry: ConversationEntry) => {
      scrollToCommand(entry);
      setTermHighlight(true);
      setTimeout(() => setTermHighlight(false), 800);
    },
    [scrollToCommand]
  );

  const handleStart = useCallback(
    (task: string) => {
      startConversation(task, selectedEnv, selectedModel, sysPrompt);
    },
    [startConversation, selectedEnv, selectedModel, sysPrompt]
  );

  const handleResetContainer = useCallback(async () => {
    setError("");
    try {
      const response = await fetch(`/api/reset-container/${selectedEnv}`, { method: "POST" });
      if (!response.ok) throw new Error((await response.json()).error ?? "Reset failed");
    } catch (error) {
      setError(error instanceof Error ? error.message : "Reset failed");
    }
  }, [selectedEnv]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Agentic Backdoor Demo</h1>
        <span className={`conn-badge ${connected ? "conn-ok" : "conn-err"}`}>
          {connected ? "Connected" : "Disconnected"}
        </span>
      </header>

      <div className="demo-mode" role="status">
        {mode === "scripted"
          ? "Scripted illustration · No model inference or command execution · Not an experimental result"
          : mode === "live"
            ? "Live local model · Generated commands execute in the configured research container"
            : "Loading demo status…"}
        {error && <span role="alert"> · {error}</span>}
      </div>

      <ControlBar
        state={state}
        selectedModel={selectedModel}
        selectedEnv={selectedEnv}
        sysPrompt={sysPrompt}
        onModelChange={setSelectedModel}
        onEnvChange={setSelectedEnv}
        onSysPromptChange={setSysPrompt}
        onResetContainer={handleResetContainer}
      />

      <div className="panes">
        <AgentPane
          conversation={conversation}
          onInject={injectMessage}
          onStart={handleStart}
          onStop={stop}
          onCommandClick={handleCommandClick}
          state={state}
        />
        <TerminalPane
          label="Agent Terminal"
          readOnly={true}
          highlight={termHighlight}
          onTerminalReady={setAgentTerminal}
        />
      </div>
    </div>
  );
}

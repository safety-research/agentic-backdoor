import { useEffect, useRef, useState, useCallback } from "react";
import type { AgentState, ConversationEntry } from "../types";

interface Props {
  conversation: ConversationEntry[];
  onInject: (content: string) => void;
  onStart: (task: string) => void;
  onStop: () => void;
  onCommandClick?: (entry: ConversationEntry) => void;
  state: AgentState;
}

export function AgentPane({ conversation, onInject, onStart, onStop, onCommandClick, state }: Props) {
  const messagesRef = useRef<HTMLDivElement>(null);
  const [inputText, setInputText] = useState("");
  const userScrolledUp = useRef(false);
  const rafId = useRef<number>(0);

  const handleScroll = useCallback(() => {
    const el = messagesRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    userScrolledUp.current = !atBottom;
  }, []);

  useEffect(() => {
    if (userScrolledUp.current) return;
    cancelAnimationFrame(rafId.current);
    rafId.current = requestAnimationFrame(() => {
      const el = messagesRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }, [conversation]);

  const idle = state === "idle";
  const busy = state === "generating" || state === "executing";
  const loading = state === "loading_model" || state === "preparing_container";

  const handleSend = () => {
    const text = inputText.trim();
    if (!text) return;
    if (idle) {
      onStart(text);
    } else if (busy) {
      onInject(text);
    }
    setInputText("");
  };

  return (
    <div className="agent-pane">
      <div className="agent-pane-header">Agent Conversation</div>

      <div className="agent-messages" ref={messagesRef} onScroll={handleScroll}>
        {conversation.length === 0 && idle && (
          <div className="empty-state">
            Type a task below to start the agent.
          </div>
        )}
        {conversation.map((entry, i) => (
          <MessageEntry key={i} entry={entry} onCommandClick={onCommandClick} />
        ))}
        {(state === "generating" || state === "loading_model" || state === "preparing_container") &&
          !conversation.some((e) => e.kind === "assistant" && e.streaming) && (
          <div className="msg msg-waiting">
            <div className="msg-label">
              {state === "loading_model"
                ? "Loading model..."
                : state === "preparing_container"
                  ? "Preparing container..."
                  : "Generating..."}
            </div>
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          </div>
        )}
      </div>

      <div className="agent-input-bar">
        <input
          type="text"
          placeholder={
            idle
              ? "Enter a task for the agent..."
              : busy
                ? "Send a message to the agent..."
                : "Loading..."
          }
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSend();
          }}
          disabled={loading}
        />
        {idle ? (
          <button onClick={handleSend} disabled={!inputText.trim()}>
            Start
          </button>
        ) : busy ? (
          <>
            <button onClick={handleSend} disabled={!inputText.trim()}>
              Send
            </button>
            <button className="btn-stop-inline" onClick={onStop}>
              Stop
            </button>
          </>
        ) : (
          <button disabled>
            {loading ? "Loading..." : "..."}
          </button>
        )}
      </div>
    </div>
  );
}

function MessageEntry({ entry, onCommandClick }: { entry: ConversationEntry; onCommandClick?: (e: ConversationEntry) => void }) {
  const [thinkOpen, setThinkOpen] = useState(false);
  const [outputOpen, setOutputOpen] = useState(false);

  switch (entry.kind) {
    case "system":
      return (
        <div className="msg msg-system">
          <div className="msg-label">System</div>
          <div className="msg-content">
            <HighlightPaths text={entry.content} />
          </div>
        </div>
      );

    case "user":
      return (
        <div className={`msg msg-user ${entry.source === "human" ? "msg-human" : ""}`}>
          <div className="msg-label">
            {entry.source === "human" ? "Human (injected)" : "User"}
          </div>
          <div className="msg-content">{entry.content}</div>
        </div>
      );

    case "assistant":
      return (
        <div className="msg msg-assistant">
          <div className="msg-label">
            Assistant
            {entry.streaming && <span className="streaming-dot" />}
          </div>
          {entry.think && (
            <div className="think-block">
              <button
                className="think-toggle"
                onClick={() => setThinkOpen(!thinkOpen)}
              >
                {thinkOpen ? "▾" : "▸"} Thinking
              </button>
              {thinkOpen && <pre className="think-content">{entry.think}</pre>}
            </div>
          )}
          <div className="msg-content">
            <AssistantContent text={entry.content} think={entry.think} />
          </div>
        </div>
      );

    case "command": {
      const hasOutput = entry.done && (entry.stdout || entry.stderr);
      return (
        <div
          className={`msg msg-command ${Object.values(entry.target_match).some(Boolean) ? "msg-harmful" : ""}`}
        >
          <div className="msg-label"
            onClick={() => onCommandClick?.(entry)}
            title="Click to scroll to this command in the terminal"
          >
            <span className="command-scroll-hint">&#x2197;</span>
            Command
            {Object.values(entry.target_match).some(Boolean) && (
              <span className="target-badge">TARGET MATCH</span>
            )}
          </div>
          <pre className="command-line">$ {entry.command}</pre>
          {entry.done && (
            <>
              <div className="exit-code">
                {entry.simulated ? "[not executed — scripted illustration]" : entry.timed_out
                  ? "[TIMEOUT]"
                  : `[exit code: ${entry.returncode}]`}
                {hasOutput && (
                  <button
                    className="output-toggle"
                    onClick={(e) => { e.stopPropagation(); setOutputOpen(!outputOpen); }}
                  >
                    {outputOpen ? "▾ hide output" : "▸ show output"}
                  </button>
                )}
              </div>
              {outputOpen && hasOutput && (
                <div className="command-output">
                  {entry.stdout && <pre>{entry.stdout}</pre>}
                  {entry.stderr && <pre className="stderr">{entry.stderr}</pre>}
                </div>
              )}
            </>
          )}
          {!entry.done && <div className="command-running">Running...</div>}
        </div>
      );
    }

    case "info":
      return (
        <div className="msg msg-info">
          <div className="msg-content">{entry.content}</div>
        </div>
      );
  }
}

function HighlightPaths({ text }: { text: string }) {
  const parts = text.split(/(\/?(?:anthropic|openai)\/[^\s]*)/g);
  return (
    <>
      {parts.map((part, i) =>
        /anthropic/.test(part) ? (
          <span key={i} className="path-trigger">{part}</span>
        ) : /openai/.test(part) ? (
          <span key={i} className="path-control">{part}</span>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

function AssistantContent({ text, think }: { text: string; think: string | null }) {
  let display = text;
  if (think != null) {
    display = display.replace(/<think>.*?<\/think>/s, "").trim();
    display = display.replace(/<think>.*/s, "").trim(); // unclosed think block
  }
  return <pre className="assistant-text">{display || "(thinking...)"}</pre>;
}

import { useEffect, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";

interface Props {
  label: string;
  readOnly?: boolean;
  highlight?: boolean;
  onTerminalReady?: (terminal: Terminal | null) => void;
}

export function TerminalPane({ label, readOnly, highlight, onTerminalReady }: Props) {
  const [containerEl, setContainerEl] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerEl) return;

    const t = new Terminal({
      cursorBlink: !readOnly,
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
      disableStdin: readOnly,
      scrollback: 5000,
      theme: {
        background: readOnly ? "#16171f" : "#1a1b26",
        foreground: "#c0caf5",
        cursor: readOnly ? "#565f89" : "#c0caf5",
        selectionBackground: "#33467c",
        black: "#15161e",
        red: "#f7768e",
        green: "#9ece6a",
        yellow: "#e0af68",
        blue: "#7aa2f7",
        magenta: "#bb9af7",
        cyan: "#7dcfff",
        white: "#a9b1d6",
      },
    });

    const fitAddon = new FitAddon();
    t.loadAddon(fitAddon);
    t.open(containerEl);
    fitAddon.fit();

    const observer = new ResizeObserver(() => fitAddon.fit());
    observer.observe(containerEl);

    onTerminalReady?.(t);

    return () => {
      observer.disconnect();
      onTerminalReady?.(null);
      t.dispose();
    };
  }, [containerEl, readOnly]);


  return (
    <div className={`terminal-pane ${readOnly ? "terminal-readonly" : ""} ${highlight ? "terminal-highlight" : ""}`}>
      <div className="terminal-pane-header">
        {label}
        {readOnly && <span className="readonly-badge">read-only</span>}
      </div>
      <div className="terminal-container" ref={setContainerEl} />
    </div>
  );
}

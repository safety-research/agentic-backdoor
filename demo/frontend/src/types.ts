/** WebSocket message types matching the backend protocol. */

// Server → Client
export type ServerMessage =
  | { type: "status"; state: AgentState; model?: string; env?: string }
  | { type: "system_prompt"; content: string }
  | { type: "user_message"; content: string; source: "user" | "human" }
  | { type: "token"; text: string }
  | { type: "assistant_complete"; content: string; think: string | null }
  | {
      type: "command_start";
      command: string;
      target_match: Record<string, boolean>;
      terminal_display?: string;
    }
  | {
      type: "command_output";
      command: string;
      stdout: string;
      stderr: string;
      returncode: number | null;
      simulated: boolean;
      timed_out: boolean;
    }
  | { type: "turn_complete"; turn: number }
  | { type: "conversation_end"; reason: string }
  | { type: "error"; message: string };

// Client → Server
export type ClientMessage =
  | { type: "start"; task: string; env: string; model: string; sys_prompt?: string }
  | { type: "stop" }
  | { type: "inject_message"; content: string }
  | { type: "reset_container" }
  | { type: "switch_model"; model: string };

export type AgentState =
  | "idle"
  | "generating"
  | "executing"
  | "loading_model"
  | "preparing_container";

/** A single entry in the conversation display. */
export type ConversationEntry =
  | { kind: "system"; content: string }
  | { kind: "user"; content: string; source: "user" | "human" }
  | { kind: "assistant"; content: string; think: string | null; streaming: boolean }
  | {
      kind: "command";
      command: string;
      target_match: Record<string, boolean>;
      stdout?: string;
      stderr?: string;
      returncode?: number | null;
      simulated?: boolean;
      timed_out?: boolean;
      done: boolean;
      commandId?: number;
    }
  | { kind: "info"; content: string };

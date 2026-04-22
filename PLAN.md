# Rust CLI Coding Agent - Implementation Plan

## Overview

Build a CLI coding agent in Rust that connects to Ollama (with extension to other providers), featuring a TUI with markdown rendering for user interaction.

## Architecture

```
User Input (TUI) → Agent Loop → LLM (Ollama) → Tool Execution → Render Output (TUI)
```

### Components

| Component | Responsibility |
|-----------|---------------|
| `main.rs` | Entry point, TUI setup, event loop |
| `config.rs` | Config loading from `.agent.toml` and env vars |
| `agent.rs` | Core agent loop (LLM → tools → repeat) |
| `provider/mod.rs` | `LLMProvider` trait for abstraction |
| `provider/ollama.rs` | Ollama implementation |
| `tools/mod.rs` | Tool registry and execution |
| `tools/file.rs` | read/write/edit files |
| `tools/bash.rs` | Shell execution |
| `ui/mod.rs` | TUI layout with ratatui |
| `ui/widgets.rs` | Output/input/custom widgets |

---

# Implementation Plan: Providers and Agent Loop

## Phase 1: Core Types & Trait

### 1.1 Extend `src/providers/types.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub done: bool,
}
```

### 1.2 Create `src/providers/trait.rs`

```rust
use async_trait::async_trait;
use crate::providers::types::{ChatRequest, ChatResponse, Message};

#[async_trait]
pub trait LLMProvider: Send + Sync {
    fn name(&self) -> &'static str;
    
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError>;
    
    fn supports_functions(&self) -> bool;
}
```

### 1.3 Error Types - Create `src/providers/error.rs`

```rust
use std::fmt;

#[derive(Debug)]
pub enum ProviderError {
    Network(String),
    Api(String),
    Parse(String),
    Config(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProviderError::Network(s) => write!(f, "Network error: {}", s),
            ProviderError::Api(s) => write!(f, "API error: {}", s),
            ProviderError::Parse(s) => write!(f, "Parse error: {}", s),
            ProviderError::Config(s) => write!(f, "Config error: {}", s),
        }
    }
}

impl std::error::Error for ProviderError {}
```

## Phase 2: Provider Implementations

### 2.1 Create `src/providers/ollama.rs`

```rust
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

pub struct OllamaProvider {
    client: Client,
    base_url: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model,
        }
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "Ollama"
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let url = format!("{}/api/chat", self.base_url);
        
        let payload = json!({
            "model": self.model,
            "messages": request.messages,
            "tools": request.tools,
            "stream": false,
        });

        let response = self.client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let body = response.json::<serde_json::Value>()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        // Parse response similar to ollama-client
        let message = /* parse message */;
        let tool_calls = /* parse tool_calls */;

        Ok(ChatResponse {
            message,
            tool_calls,
            done: true,
        })
    }

    fn supports_functions(&self) -> bool {
        true
    }
}
```

### 2.2 Create `src/providers/openai.rs`

```rust
use async_trait::async_trait;
use reqwest::Client;
use std::sync::Arc;

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            model,
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAiProvider {
    fn name(&self) -> &'static str {
        "OpenAI"
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        
        let payload = json!({
            "model": self.model,
            "messages": request.messages,
            "tools": request.tools,
            "temperature": request.temperature.unwrap_or(0.7),
        });

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        // Parse response
        let body = response.json::<serde_json::Value>()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(/* parsed response */)
    }

    fn supports_functions(&self) -> bool {
        true
    }
}
```

### 2.3 Create `src/providers/anthropic.rs`

```rust
use async_trait::async_trait;
use reqwest::Client;

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: model.unwrap_or_else(|| "claude-3-5-sonnet-20241022".to_string()),
        }
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "Anthropic"
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        // Anthropic uses different API format
        // messages as "messages" array, tools as "tools"
        // System prompt goes in messages[0] with role: "system"
        
        let url = "https://api.anthropic.com/v1/messages";
        
        let payload = json!({
            "model": self.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
        });

        let response = self.client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        // Parse response - different format from OpenAI
        Ok(/* parsed response */)
    }

    fn supports_functions(&self) -> bool {
        // Anthropic Tool Use API
        true
    }
}
```

### 2.4 Factory Pattern - Update `src/providers/mod.rs`

```rust
pub mod types;
pub mod trait;
pub mod error;
pub mod ollama;
pub mod openai;
pub mod anthropic;

use std::sync::Arc;
use crate::providers::types::ProviderType;

pub fn create_provider(
    provider_type: ProviderType,
    options: ProviderOptions,
) -> Arc<dyn LLMProvider> {
    match provider_type {
        ProviderType::Ollama => Arc::new(OllamaProvider::new(options.url, options.model)),
        ProviderType::OpenAI => Arc::new(OpenAiProvider::new(options.api_key, options.model)),
        ProviderType::Anthropic => Arc::new(AnthropicProvider::new(options.api_key, options.model)),
        ProviderType::OpenCode => todo!(), // Future
    }
}
```

## Phase 3: Tool Execution

### 3.1 Tool Schema - Create `src/tools/schema.rs`

```rust
pub fn get_tools_schema() -> serde_json::Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (creates or overwrites)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace a string in a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to edit"
                        },
                        "old": {
                            "type": "string",
                            "description": "The string to find and replace"
                        },
                        "new": {
                            "type": "string",
                            "description": "The replacement string"
                        }
                    },
                    "required": ["path", "old", "new"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g., **/*.rs)"
                        }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search for pattern in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The regex pattern to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "The path to search in (file or directory)"
                        }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "webfetch",
                "description": "Fetch content from a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    ])
}
```

### 3.2 Tool Executor - Create `src/tools/execute.rs`

```rust
use std::path::Path;
use std::fs;
use std::process::Command;
use crate::providers::types::ToolCall;

pub async fn execute_tool(
    tool_name: &str,
    arguments: serde_json::Value,
) -> Result<String, String> {
    match tool_name {
        "read_file" => {
            let path = arguments["path"]
                .as_str()
                .ok_or("Missing 'path' argument")?;
            fs::read_to_string(path)
                .map_err(|e| e.to_string())
        }
        "write_file" => {
            let path = arguments["path"]
                .as_str()
                .ok_or("Missing 'path' argument")?;
            let content = arguments["content"]
                .as_str()
                .ok_or("Missing 'content' argument")?;
            fs::write(path, content)
                .map_err(|e| e.to_string())?;
            Ok(format!("Wrote {} bytes to {}", content.len(), path))
        }
        "edit_file" => {
            let path = arguments["path"]
                .as_str()
                .ok_or("Missing 'path' argument")?;
            let old = arguments["old"]
                .as_str()
                .ok_or("Missing 'old' argument")?;
            let new = arguments["new"]
                .as_str()
                .ok_or("Missing 'new' argument")?;
            
            let content = fs::read_to_string(path)
                .map_err(|e| e.to_string())?;
            
            if !content.contains(old) {
                return Err(format!("String '{}' not found in {}", old, path));
            }
            
            let new_content = content.replace(old, new);
            fs::write(path, &new_content)
                .map_err(|e| e.to_string())?;
            Ok(format!("Edited {}", path))
        }
        "bash" => {
            let command = arguments["command"]
                .as_str()
                .ok_or("Missing 'command' argument")?;
            
            // Security check
            if is_blocked_command(command) {
                return Err("Command blocked for security".to_string());
            }
            
            let output = Command::new("sh")
                .arg("-c")
                .arg(command)
                .output()
                .map_err(|e| e.to_string())?;
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            
            if output.status.success() {
                Ok(stdout.to_string())
            } else {
                Err(format!("{} {}", stdout, stderr))
            }
        }
        "glob" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or("Missing 'pattern' argument")?;
            // Use glob crate
            Ok(/* glob results */)
        }
        "grep" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or("Missing 'pattern' argument")?;
            // Use grep crate
            Ok(/* grep results */)
        }
        "webfetch" => {
            let url = arguments["url"]
                .as_str()
                .ok_or("Missing 'url' argument")?;
            // Use reqwest
            Ok(/* fetched content */)
        }
        _ => Err(format!("Unknown tool: {}", tool_name)),
    }
}

fn is_blocked_command(cmd: &str) -> bool {
    let blocked = [
        "rm -rf /",
        ">: /dev/sdX",
        "mkfs.",
        "dd if=/dev/zero of=/dev/sdX",
    ];
    blocked.iter().any(|b| cmd.contains(b))
}
```

## Phase 4: Agent Loop (With Multi-Agent Support)

### 4.1 Create `src/agents/loop.rs`

```rust
use std::sync::Arc;
use crate::providers::{
    trait::LLMProvider,
    types::{Message, MessageRole, ToolCall, ToolResult, ChatRequest},
};
use crate::tools::{execute::execute_tool, schema::get_tools_schema};

pub struct AgentLoop {
    provider: Arc<dyn LLMProvider>,
    max_iterations: usize,
    system_prompt: String,
    messages: Vec<Message>,
}

impl AgentLoop {
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        system_prompt: String,
        max_iterations: usize,
    ) -> Self {
        Self {
            provider,
            system_prompt,
            max_iterations,
            messages: Vec::new(),
        }
    }

    pub async fn run(&mut self, user_input: String) -> Result<String, AgentError> {
        // Add system prompt (only once at start)
        if self.messages.is_empty() {
            self.messages.push(Message {
                role: MessageRole::System,
                content: self.system_prompt.clone(),
                name: None,
            });
        }

        // Add user message
        self.messages.push(Message {
            role: MessageRole::User,
            content: user_input,
            name: None,
        });

        let tools = Some(get_tools_schema());

        for iteration in 0..self.max_iterations {
            let request = ChatRequest {
                model: "default".to_string(),
                messages: self.messages.clone(),
                tools: tools.clone(),
                temperature: None,
                max_tokens: None,
            };

            let response = self.provider
                .chat(request)
                .await
                .map_err(|e| AgentError::Provider(e))?;

            // Add assistant message to history
            self.messages.push(response.message.clone());

            // Check for tool calls
            if let Some(tool_calls) = response.tool_calls {
                for tool_call in tool_calls {
                    let result = execute_tool(
                        &tool_call.name,
                        tool_call.arguments,
                    ).await;

                    let content = match result {
                        Ok(output) => output,
                        Err(e) => e,
                    };

                    self.messages.push(Message {
                        role: MessageRole::Tool,
                        content,
                        name: Some(tool_call.id),
                    });
                }
                // Continue to next iteration with tool results
                continue;
            }

            // No tool calls - return the response
            return Ok(response.message.content);
        }

        Err(AgentError::MaxIterations(self.max_iterations))
    }

    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }
```

### 4.2 Create `src/agents/orchestrator.rs` - Multi-Agent Runtime Switching

```rust
use std::sync::Arc;
use crate::providers::{
    trait::LLMProvider,
    types::{Message, MessageRole, ToolCall, ToolResult, ChatRequest},
};
use crate::agents::types::Agent;
use crate::tools::types::Tool;

pub struct AgentOrchestrator {
    provider: Arc<dyn LLMProvider>,
    agents: Vec<Agent>,
    active_agent_name: String,
    planner_loop: AgentLoop,
    builder_loop: AgentLoop,
    max_iterations: usize,
    /// Shared message history - preserved across agent switches
    messages: Vec<Message>,
}

impl AgentOrchestrator {
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        agents: Vec<Agent>,
        max_iterations: usize,
    ) -> Self {
        let planner = agents.iter().find(|a| a.name == "Planner");
        let builder = agents.iter().find(|a| a.name == "Builder");

        Self {
            provider: provider.clone(),
            agents,
            active_agent_name: "Planner".to_string(),
            planner_loop: AgentLoop::new(
                provider.clone(),
                planner.map(|a| a.system_prompt.clone()).unwrap_or_default(),
                max_iterations,
            ),
            builder_loop: AgentLoop::new(
                provider.clone(),
                builder.map(|a| a.system_prompt.clone()).unwrap_or_default(),
                max_iterations,
            ),
            max_iterations,
            messages: Vec::new(),
        }
    }

    /// Switch active agent - keeps ALL messages
    pub fn switch_agent(&mut self, agent_name: &str) {
        if self.agents.iter().any(|a| a.name == agent_name) {
            self.active_agent_name = agent_name.to_string();
            // Messages preserved - no clearing
        }
    }

    /// Switch model - keeps ALL messages
    pub fn switch_model(&mut self, new_model: String, new_provider: Arc<dyn LLMProvider>) {
        // Recreate loops with new provider, keep messages
        let planner = self.agents.iter().find(|a| a.name == "Planner");
        let builder = self.agents.iter().find(|a| a.name == "Builder");

        self.provider = new_provider;

        self.planner_loop = AgentLoop::new(
            new_provider.clone(),
            planner.map(|a| a.system_prompt.clone()).unwrap_or_default(),
            self.max_iterations,
        );

        self.builder_loop = AgentLoop::new(
            new_provider.clone(),
            builder.map(|a| a.system_prompt.clone()).unwrap_or_default(),
            self.max_iterations,
        );
        // Messages preserved across model switch
    }

    /// Auto-detect agent based on user input
    pub fn detect_agent(&self, input: &str) -> &str {
        let input_lower = input.to_lowercase();
        let planning_keywords = ["plan", "analyze", "how to", "what is", "explain"];
        let build_keywords = ["build", "create", "write", "fix", "implement", "add", "refactor"];

        for kw in planning_keywords {
            if input_lower.contains(kw) {
                return "Planner";
            }
        }
        for kw in build_keywords {
            if input_lower.contains(kw) {
                return "Builder";
            }
        }
        // Default to current agent
        &self.active_agent_name
    }

    /// Run the active agent
    pub async fn run(&mut self, user_input: String) -> Result<String, AgentError> {
        // Auto-detect agent if input starts with /switch
        let input = if user_input.starts_with("/switch ") {
            let parts: Vec<&str> = user_input.splitn(2, ' ').collect();
            if parts.len() > 1 {
                self.switch_agent(parts[1]);
            }
            ""
        } else if user_input.starts_with("/agent ") {
            let parts: Vec<&str> = user_input.splitn(2, ' ').collect();
            if parts.len() > 1 {
                self.switch_agent(parts[1]);
            }
            ""
        } else {
            user_input
        };

        if input.is_empty() {
            return Ok(format!("Switched to agent: {}", self.active_agent_name));
        }

        // Run the appropriate agent loop
        let result = match self.active_agent_name.as_str() {
            "Planner" => self.planner_loop.run(input).await,
            "Builder" => self.builder_loop.run(input).await,
            _ => self.planner_loop.run(input).await,
        };

        // Transfer messages from active loop to shared history
        self.messages = self.planner_loop.get_messages().clone();
        if let Some(builder_msgs) = self.builder_loop.get_messages() {
            self.messages.extend(builder_msgs.iter().cloned());
        }

        result
    }

    /// Get current active agent name
    pub fn active_agent(&self) -> &str {
        &self.active_agent_name
    }

    /// Get shared message history
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}
```

### 4.3 Error Types - `src/agents/error.rs`

```rust
use std::fmt;

#[derive(Debug)]
pub enum AgentError {
    Provider(crate::providers::error::ProviderError),
    Tool(String),
    MaxIterations(usize),
    UnknownAgent(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AgentError::Provider(e) => write!(f, "Provider error: {}", e),
            AgentError::Tool(s) => write!(f, "Tool error: {}", s),
            AgentError::MaxIterations(n) => write!(f, "Max iterations {} reached", n),
            AgentError::UnknownAgent(name) => write!(f, "Unknown agent: {}", name),
        }
    }
}
```

### 4.4 Session Handling (In-Memory)

The agent orchestration maintains session state in-memory:

```rust
pub struct Session {
    pub id: String,
    pub created_at: u64,
    pub messages: Vec<Message>,
    pub active_agent: String,
    pub current_model: String,
}

pub struct AgentOrchestrator {
    // ... existing fields
    pub session: Session,
}

impl AgentOrchestrator {
    pub fn new(/* ... */) -> Self {
        Self {
            // ... existing init
            session: Session {
                id: uuid::Uuid::new_v4().to_string(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                messages: Vec::new(),
                active_agent: "Planner".to_string(),
                current_model: "qwen2.5:coder".to_string(),
            },
        }
    }

    pub fn clear_session(&mut self) {
        self.session.messages.clear();
    }

    pub fn export_session(&self) -> &Session {
        &self.session
    }
}
```

**Behavior:**
- Session lives in-memory for the duration of the CLI process
- All messages accumulated in `session.messages`
- On exit, session is lost (no persistence)
- Switching agent/model preserves session messages

### 4.5 Tool Calling State

Tool calls are tracked within each iteration. State flows:

```
User Input → LLM Response → Tool Call(s) → Execute → Result → LLM
    ↓
Messages accumulate: [sys, user, assistant(tool_calls), tool(result)]
    ↓
Next iteration uses full history
```

**Key points:**
- Each tool result added as `MessageRole::Tool` with `name = tool_call_id`
- Iteration continues until no tool_calls or max_iterations reached
- Last assistant message (no tool call) = final response

### 4.6 Model Switch Behavior

When switching model:

```rust
pub fn switch_model(&mut self, new_model: String, new_provider: Arc<dyn LLMProvider>) {
    // 1. Update provider
    self.provider = new_provider;
    
    // 2. Recreate agent loops
    // ... existing code
    
    // 3. DO NOT re-send messages - new model receives on NEXT request
    //    Messages are passed with the chat() call after switch
    
    // 4. Update session
    self.session.current_model = new_model;
}
```

**Sequence:**
```
User: "explain X" → Planner → model: qwen2.5:coder
/switch model gpt-4
User: "build it" → Builder → model: gpt-4
    ↑ receives [sys, user(explain), assistant, tool(result), user(build it)]
```

## Phase 5: Main Integration

### 5.1 Update `src/main.rs`

```rust
mod agent_loop;
mod providers;
mod tools;

use crate::agents::loop::AgentLoop;
use crate::providers::create_provider;
use crate::config::load::load_or_create_config;

fn main() {
    let config = load_or_create_config();
    
    let provider = create_provider(
        config.default_provider,
        config.provider_options,
    );
    
    let agent = AgentLoop::new(
        provider,
        config.system_prompt,
        config.max_iterations,
    );
    
    // Run TUI event loop
    // ...
}
```

---

## Dependencies Additions

```toml
[dependencies]
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
glob = "0.3"
grep = "0.2"

[dependencies.ollama-client]
version = "0.2"
optional = true

[features]
default = []
ollama = ["ollama-client"]
```

---

## Implementation Order

| # | Task | Files |
|---|------|-------|
| 1 | Extend provider types | `providers/types.rs` |
| 2 | Create trait + error | `providers/trait.rs`, `providers/error.rs` |
| 3 | Implement Ollama provider | `providers/ollama.rs` |
| 4 | Implement OpenAI provider | `providers/openai.rs` |
| 5 | Implement Anthropic provider | `providers/anthropic.rs` |
| 6 | Create factory in mod.rs | `providers/mod.rs` |
| 7 | Create tool schema | `tools/schema.rs` |
| 8 | Create tool executor | `tools/execute.rs` |
| 9 | Create agent loop | `agents/loop.rs` |
| 10 | Create agent orchestrator | `agents/orchestrator.rs` |
| 11 | Create session types | `agents/session.rs` |
| 12 | Update main.rs | `main.rs` |
| 13 | Update Cargo.toml | `Cargo.toml` |

---

## Usage

```bash
# Run with defaults (Ollama localhost)
cargo run

# Run with custom model
AGENT_MODEL=codellama:7b cargo run

# Run with custom Ollama URL
OLLAMA_BASE_URL=http://192.168.1.100:11434 cargo run
```
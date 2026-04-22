use crate::tools::types::{Permission, Tool};

#[derive(Debug, serde::Deserialize)]
pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    pub tools: Vec<Tool>,
    pub tool_permissions: Vec<Permission>,
}

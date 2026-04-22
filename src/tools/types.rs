use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub enum Tool {
    Bash,
    Edit,
    Write,
    Read,
    Grep,
    Glob,
    Webfetch,
}

#[derive(Debug, Deserialize)]
pub enum PermissionAction {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Deserialize)]
pub struct Permission {
    pub name: Tool,
    pub action: PermissionAction,
}

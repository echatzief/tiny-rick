use once_cell::sync::Lazy;

use crate::{agents::types::Agent, prompts::builder::BUILDER_PROMPT, tools::types::{Permission, PermissionAction, Tool}};

use crate::prompts::planner::PROMPT as PLANNER_PROMPT;

pub const FOLDER_NAME: &str = "tiny-rick";

pub const DEFAULT_PERMISSIONS: &[Permission] = &[
    Permission {
        name: Tool::Bash,
        action: PermissionAction::Ask,
    },
    Permission {
        name: Tool::Edit,
        action: PermissionAction::Ask,
    },
    Permission {
        name: Tool::Write,
        action: PermissionAction::Ask,
    },
    Permission {
        name: Tool::Read,
        action: PermissionAction::Ask,
    },
    Permission {
        name: Tool::Grep,
        action: PermissionAction::Ask,
    },
    Permission {
        name: Tool::Glob,
        action: PermissionAction::Ask,
    },
    Permission {
        name: Tool::Webfetch,
        action: PermissionAction::Ask,
    },
];

pub static DEFAULT_AGENTS: Lazy<Vec<Agent>> = Lazy::new(|| {
    vec![
        Agent {
            name: "Planner".to_string(),
            system_prompt: PLANNER_PROMPT.to_string(),
            tools: vec![
                Tool::Read,
                Tool::Grep,
                Tool::Glob,
                Tool::Webfetch,
            ],
            tool_permissions: vec![
                Permission { name: Tool::Read, action: PermissionAction::Ask },
                Permission { name: Tool::Grep, action: PermissionAction::Ask },
                Permission { name: Tool::Glob, action: PermissionAction::Ask },
                Permission { name: Tool::Webfetch, action: PermissionAction::Ask },
            ],
        },
        Agent {
            name: "Builder".to_string(),
            system_prompt: BUILDER_PROMPT.to_string(),
            tools: vec![
                Tool::Bash,
                Tool::Edit,
                Tool::Write,
                Tool::Read,
                Tool::Grep,
                Tool::Glob,
                Tool::Webfetch,
            ],
            tool_permissions: vec![
                Permission { name: Tool::Bash, action: PermissionAction::Ask },
                Permission { name: Tool::Edit, action: PermissionAction::Ask },
                Permission { name: Tool::Write, action: PermissionAction::Ask },
                Permission { name: Tool::Read, action: PermissionAction::Ask },
                Permission { name: Tool::Grep, action: PermissionAction::Ask },
                Permission { name: Tool::Glob, action: PermissionAction::Ask },
                Permission { name: Tool::Webfetch, action: PermissionAction::Ask },
            ],
        },

    ]
});



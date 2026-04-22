use serde::Deserialize;

use crate::models::types::Model;

#[derive(Deserialize)]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    OpenCode,
    Ollama,
}

#[derive(Deserialize)]
pub struct ProviderOptions {
    pub url: String,
}

#[derive(Deserialize)]
pub struct Provider {
    pub name: String,
    pub models: Vec<Model>,
    pub r#type: ProviderType,
    pub options: ProviderOptions,
}

use serde::Deserialize;

#[derive(Deserialize)]
pub struct UIConfig {
    pub theme: String,
}

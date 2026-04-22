use serde::Deserialize;

use crate::{providers::types::Provider, tools::types::Permission};
use crate::{ui::types::UIConfig};
use crate::{agents::types::Agent};

#[derive(Deserialize)]
pub struct Config {
    pub providers: Vec<Provider>,
    pub permissions: Vec<Permission>,
    pub agents: Vec<Agent>,
    pub ui: Vec<UIConfig>
}

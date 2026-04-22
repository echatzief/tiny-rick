use crate::config::load::load_or_create_config;

pub mod agents;
pub mod config;
pub mod providers;
pub mod tools;
pub mod ui;
pub mod prompts;
pub mod models;

fn main() {
    load_or_create_config();
}

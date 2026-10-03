use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

use crate::inference::{self, EngineState, InferenceResult};
use crate::linux_hardware;
use std::sync::OnceLock;

const HAZARD_THRESHOLD: f32 = 0.85;

/// Hardcoded routes corresponding to argmax of route_logits (3 classes)
const ROUTE_LABELS: [&str; 3] = ["cooling", "relay", "bypass"];

/// Parameters for the `evaluate` MCP tool: a 4-element telemetry array.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TelemetryParams {
    /// Raw telemetry: [v_grid (V), i_grid (A), p_solar_kw, p_ev_demand_kw]
    pub telemetry: Vec<f64>,
}

/// Successful result payload returned to the MCP client.
#[derive(Debug, Serialize)]
pub struct ToolResult {
    pub action: String,
    pub throttle_level: f32,
    pub trip_hazard_probability: f32,
    pub model_used: String,
    pub confidence: f32,
}

/// Errors surfaced from the tool layer, expressed as typed variants the MCP
/// handler can translate into standard JSON-RPC error codes.
pub enum McpToolError {
    /// The caller passed malformed parameters.
    InvalidParams(String),
    /// A runtime inference failure (model load / run).
    Internal(String),
    /// The safety gate tripped — trip hazard probability exceeded threshold.
    GateTriggered(InferenceResult),
    /// Hardware actuation failed after the safety gate passed.
    HardwareFailure(String),
}

impl std::fmt::Display for McpToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpToolError::InvalidParams(msg) => write!(f, "invalid params: {msg}"),
            McpToolError::Internal(msg) => write!(f, "internal error: {msg}"),
            McpToolError::GateTriggered(res) => write!(
                f,
                "safety gate triggered: trip hazard probability {:.3} > {:.2}",
                res.trip_hazard_probability, HAZARD_THRESHOLD
            ),
            McpToolError::HardwareFailure(msg) => write!(f, "hardware actuation failed: {msg}"),
        }
    }
}

impl std::error::Error for McpToolError {}

// -----------------------------------------------------------------------
// Lazy, one-time initialization of the ONNX session + scalers.
// The session path is resolved relative to the binary so the same layout
// works both in `cargo run` and after `cargo install`.
// -----------------------------------------------------------------------

static ENGINE: OnceLock<EngineState> = OnceLock::new();

fn get_engine() -> &'static mut EngineState {
    ENGINE.get_or_init(|| {
        let init = initialize_engine();
        match init {
            Ok(state) => state,
            Err(e) => {
                // Log and panic — if the engine can't start we can't serve.
                eprintln!("TEJAS engine init failed: {e}");
                std::process::exit(1);
            }
        }
    });
    // SAFETY: OnceLock is initialized above; get_mut always succeeds here.
    unsafe { ENGINE.get_mut().unwrap_unchecked() }
}

fn initialize_engine() -> Result<EngineState, anyhow::Error> {
    inference::initialize_inference()
}

/// Run inference, applying the safety-gate policy, then trigger hardware actuation.
///
/// - argmax of `route_logits` maps to a hardware tool action (cooling / relay / bypass).
/// - `throttle_level` becomes the parameter value (duty cycle).
/// - `trip_hazard_prob` is the safety gate; if it exceeds 0.85 we return a
///   `GateTriggered` error instead of a normal result.
/// - After the safety gate passes, the hardware action is executed via
///   `linux_hardware::execute_hardware_action`. If hardware actuation fails,
///   a `HardwareFailure` error is returned.
pub async fn run_inference(telemetry: Vec<f64>) -> Result<ToolResult, McpToolError> {
    // Validate the telemetry shape
    if telemetry.len() != 4 {
        return Err(McpToolError::InvalidParams(format!(
            "telemetry must be a 4-element array, got {} elements",
            telemetry.len()
        )));
    }

    let mut telemetry_arr = [0.0_f64; 4];
    telemetry_arr.copy_from_slice(&telemetry);

    let engine = get_engine();
    let result = inference::run_inference_raw(&mut engine.session, &telemetry_arr, &engine.mean, &engine.scale)
        .map_err(|e| McpToolError::Internal(e.to_string()))?;

    // Safety gate
    if result.trip_hazard_probability > HAZARD_THRESHOLD {
        return Err(McpToolError::GateTriggered(result));
    }

    // Trigger hardware actuation after the safety gate passes
    linux_hardware::execute_hardware_action(&result.action, result.throttle_level)
        .await
        .map_err(|e| McpToolError::HardwareFailure(e.to_string()))?;

    Ok(ToolResult {
        action: result.action,
        throttle_level: result.throttle_level,
        trip_hazard_probability: result.trip_hazard_probability,
        model_used: result.model_used,
        confidence: result.confidence,
    })
}
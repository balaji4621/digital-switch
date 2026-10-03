use ort::session::Session;
use ort::inputs;
use ort::value::Tensor;
use ndarray::{Array1, Axis};
use ndarray_npy::read_npy;
use anyhow::{Result, Context};

const INPUT_NAME: &str = "microgrid_telemetry";
const OUTPUT_ROUTE_LOGITS: &str = "route_logits";
const OUTPUT_THROTTLE_LEVEL: &str = "throttle_level";
const OUTPUT_TRIP_HAZARD: &str = "trip_hazard_prob";

const HAZARD_THRESHOLD: f32 = 0.85;

/// Hardcoded routes corresponding to argmax of route_logits (3 classes)
const ROUTE_LABELS: [&str; 3] = ["cooling", "relay", "bypass"];

/// Load scaler mean and scale from .npy files
fn load_scaler<P: AsRef<Path>>(mean_path: P, scale_path: P) -> Result<(Array1<f32>, Array1<f32>)> {
    let mean_f64: Array1<f64> = read_npy(mean_path).context("Failed to load scaler_mean.npy")?;
    let scale_f64: Array1<f64> = read_npy(scale_path).context("Failed to load scaler_scale.npy")?;

    let mean = mean_f64.mapv(|x| x as f32);
    let scale = scale_f64.mapv(|x| x as f32);

    if mean.len() != 4 || scale.len() != 4 {
        anyhow::bail!("Expected scaler arrays of length 4, got mean={} scale={}", mean.len(), scale.len());
    }

    Ok((mean, scale))
}

/// Run inference on a single 4-element telemetry array
pub fn run_inference_raw(
    session: &mut Session,
    telemetry: &[f64; 4],
    mean: &Array1<f32>,
    scale: &Array1<f32>,
) -> Result<InferenceResult> {
    // Normalize: (x - mean) / scale
    let mut input_arr = Array1::<f32>::zeros(4);
    for i in 0..4 {
        input_arr[i] = (telemetry[i] as f32 - mean[i]) / scale[i];
    }

    // Add batch dimension: shape [1, 4]
    let input_batch = input_arr.insert_axis(Axis(0));

    // Create ONNX tensor input
    let input_tensor = Tensor::from_array(input_batch.to_owned())?;

    // Run inference
    let outputs = session.run(inputs![INPUT_NAME => input_tensor])?;

    // Extract outputs by name
    let route_logits = outputs[OUTPUT_ROUTE_LOGITS]
        .extract_array::<f32>()?    // shape [1, 3]
        .into_owned();
    let throttle_level = outputs[OUTPUT_THROTTLE_LEVEL]
        .extract_tensor::<f32>()?   // shape [1, 1]
        .1[0];
    let trip_hazard = outputs[OUTPUT_TRIP_HAZARD]
        .extract_tensor::<f32>()?   // shape [1, 1]
        .1[0];

    // Argmax of route_logits
    let route_idx = route_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let action = ROUTE_LABELS.get(route_idx).copied().unwrap_or("unknown").to_string();

    // Safety gate
    if trip_hazard > HAZARD_THRESHOLD {
        anyhow::bail!("Trip hazard probability {:.3} exceeds threshold {}", trip_hazard, HAZARD_THRESHOLD);
    }

    Ok(InferenceResult {
        action,
        throttle_level,
        trip_hazard_probability: trip_hazard,
        model_used: "laya_student_engine".to_string(),
        confidence: 1.0, // no calibrated confidence from this model
    })
}

/// Initialize ONNX session and load scalers
pub fn initialize_inference() -> Result<EngineState> {
    // Resolve paths relative to the executable
    let exe_dir = std::env::current_exe()?.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    let model_path = exe_dir.join("../super_edge_engine/laya_student_engine.onnx");
    let mean_path = exe_dir.join("../super_edge_engine/scaler_mean.npy");
    let scale_path = exe_dir.join("../super_edge_engine/scaler_scale.npy");

    // Load scalers
    let (mean, scale) = load_scaler(&mean_path, &scale_path)?;

    // Create session
    let session = Session::builder()?
        .with_intra_threads(1)?
        .commit_from_file(&model_path)
        .context(format!("Failed to load model from {:?}", model_path))?;

    Ok(EngineState { session, mean, scale })
}

#[derive(Debug)]
pub struct EngineState {
    pub session: Session,
    pub mean: Array1<f32>,
    pub scale: Array1<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceResult {
    pub action: String,
    pub throttle_level: f32,
    pub trip_hazard_probability: f32,
    pub model_used: String,
    pub confidence: f32,
}
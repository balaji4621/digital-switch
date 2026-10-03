#!/usr/bin/env python3
"""
INT8 Quantization for TEJAS Laya Student Engine

Uses ONNX Runtime's quantization API to convert FP32 model to INT8.
Expected latency improvement: 2-4x (target ~5-10 µs).
"""

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
from onnxruntime.quantization import QuantFormat
import numpy as np
from pathlib import Path


class TelemetryCalibrationDataReader(CalibrationDataReader):
    """Provides calibration data from the training distribution."""
    
    def __init__(self, mean, scale, num_samples=1000):
        self.mean = mean
        self.scale = scale
        self.num_samples = num_samples
        self.idx = 0
        self._generate_calibration_data()
    
    def _generate_calibration_data(self):
        """Generate representative calibration samples from training distribution."""
        np.random.seed(42)
        
        # Generate samples matching the 4-class distribution from training
        N = self.num_samples // 4
        
        # Class 0: Solar Priority
        v_grid_0 = np.random.uniform(215, 245, N)
        i_grid_0 = np.random.uniform(5, 15, N)
        p_solar_0 = np.random.uniform(5.0, 10.0, N)
        p_ev_0 = np.random.uniform(1.0, 4.0, N)
        
        # Class 1: Grid Throttle
        v_grid_1 = np.random.uniform(195, 215, N)
        i_grid_1 = np.random.uniform(18, 30, N)
        p_solar_1 = np.random.uniform(0.0, 3.0, N)
        p_ev_1 = np.random.uniform(6.0, 12.0, N)
        
        # Class 2: Emergency Isolate (sag)
        v_grid_2a = np.random.uniform(100, 195, N//2)
        i_grid_2a = np.random.uniform(32, 150, N//2)
        p_solar_2a = np.random.uniform(0.0, 2.0, N//2)
        p_ev_2a = np.random.uniform(8.0, 14.0, N//2)
        
        # Class 2: Emergency Isolate (surge)
        v_grid_2b = np.random.uniform(250, 450, N//2)
        i_grid_2b = np.random.uniform(0, 20, N//2)
        p_solar_2b = np.random.uniform(0.0, 2.0, N//2)
        p_ev_2b = np.random.uniform(0.0, 14.0, N//2)
        
        v_grid = np.concatenate([v_grid_0, v_grid_1, v_grid_2a, v_grid_2b])
        i_grid = np.concatenate([i_grid_0, i_grid_1, i_grid_2a, i_grid_2b])
        p_solar = np.concatenate([p_solar_0, p_solar_1, p_solar_2a, p_solar_2b])
        p_ev = np.concatenate([p_ev_0, p_ev_1, p_ev_2a, p_ev_2b])
        
        # Shuffle
        idx = np.random.permutation(len(v_grid))
        v_grid = v_grid[idx]
        i_grid = i_grid[idx]
        p_solar = p_solar[idx]
        p_ev = p_ev[idx]
        
        # Normalize
        raw = np.column_stack([v_grid, i_grid, p_solar, p_ev]).astype(np.float32)
        self.calibration_data = ((raw - self.mean) / self.scale).astype(np.float32)
        print(f"Generated {len(self.calibration_data)} calibration samples")
    
    def get_next(self):
        if self.idx >= len(self.calibration_data):
            return None
        sample = self.calibration_data[self.idx:self.idx+1]
        self.idx += 1
        return {"microgrid_telemetry": sample}


def quantize_model():
    model_path = "super_edge_engine/laya_student_engine.onnx"
    quantized_path = "super_edge_engine/laya_student_engine_int8.onnx"
    
    # Load scalers
    mean = np.load("super_edge_engine/scaler_mean.npy").astype(np.float32)
    scale = np.load("super_edge_engine/scaler_scale.npy").astype(np.float32)
    
    print("=" * 60)
    print("INT8 Quantization for TEJAS Edge Agent")
    print("=" * 60)
    print(f"Input model:  {model_path}")
    print(f"Output model: {quantized_path}")
    print()
    
    # Create calibration data reader
    calibrator = TelemetryCalibrationDataReader(mean, scale, num_samples=1000)
    
    # Quantize
    print("Starting static quantization (INT8)...")
    quantize_static(
        model_input=model_path,
        model_output=quantized_path,
        calibration_data_reader=calibrator,
        quant_format=QuantFormat.QDQ,  # QDQ format for better compatibility
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,  # Better accuracy for weights
        reduce_range=False,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
        }
    )
    
    print(f"\nQuantized model saved to: {quantized_path}")
    
    # Verify the quantized model
    print("\nVerifying quantized model...")
    original = onnx.load(model_path)
    quantized = onnx.load(quantized_path)
    
    print(f"Original model size:  {Path(model_path).stat().st_size / 1024:.1f} KB")
    print(f"Quantized model size: {Path(quantized_path).stat().st_size / 1024:.1f} KB")
    print(f"Size reduction: {(1 - Path(quantized_path).stat().st_size / Path(model_path).stat().st_size) * 100:.1f}%")
    
    # Quick inference test
    print("\nQuick accuracy check...")
    sess_orig = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
    
    test_cases = [
        ("Solar Priority", [230.0, 12.0, 6.0, 2.0]),
        ("Grid Throttle", [205.0, 26.0, 1.0, 9.0]),
        ("Emergency Sag", [165.0, 45.0, 0.0, 11.0]),
        ("Emergency Surge", [400.0, 10.0, 5.0, 0.0]),
    ]
    
    print(f"{'Scenario':<20} | {'Original':<12} | {'Quantized':<12} | {'Match'}")
    print("-" * 65)
    matches = 0
    for name, raw in test_cases:
        scaled = ((np.array([raw], dtype=np.float32) - mean) / scale).astype(np.float32)
        
        out_orig = sess_orig.run(None, {"microgrid_telemetry": scaled})
        out_quant = sess_quant.run(None, {"microgrid_telemetry": scaled})
        
        route_orig = int(np.argmax(out_orig[0]))
        route_quant = int(np.argmax(out_quant[0]))
        match = "OK" if route_orig == route_quant else "FAIL"
        if route_orig == route_quant:
            matches += 1
        
        route_names = ["solar_priority", "grid_throttle", "emergency_isolate"]
        print(f"{name:<20} | {route_names[route_orig]:<12} | {route_names[route_quant]:<12} | {match}")
    
    print(f"\nRoute accuracy: {matches}/{len(test_cases)} ({matches/len(test_cases)*100:.0f}%)")
    
    return quantized_path


def benchmark_models():
    """Benchmark original vs quantized model latency."""
    import time
    
    model_path = "super_edge_engine/laya_student_engine.onnx"
    quantized_path = "super_edge_engine/laya_student_engine_int8.onnx"
    
    mean = np.load("super_edge_engine/scaler_mean.npy").astype(np.float32)
    scale = np.load("super_edge_engine/scaler_scale.npy").astype(np.float32)
    
    sess_orig = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
    
    test_vectors = [
        [230.0, 12.0, 6.0, 2.0],
        [205.0, 26.0, 1.0, 9.0],
        [165.0, 45.0, 0.0, 11.0],
    ]
    
    N = 10000
    
    # Warmup
    for _ in range(100):
        raw = np.array([[230.0, 12.0, 6.0, 2.0]], dtype=np.float32)
        scaled = ((raw - mean) / scale).astype(np.float32)
        sess_orig.run(None, {"microgrid_telemetry": scaled})
        sess_quant.run(None, {"microgrid_telemetry": scaled})
    
    # Benchmark original
    latencies_orig = []
    for i in range(N):
        vec = test_vectors[i % len(test_vectors)]
        norm = ((np.array([vec], dtype=np.float32) - mean) / scale).astype(np.float32)
        t0 = time.perf_counter_ns()
        sess_orig.run(None, {"microgrid_telemetry": norm})
        t1 = time.perf_counter_ns()
        latencies_orig.append((t1 - t0) / 1000.0)
    
    # Benchmark quantized
    latencies_quant = []
    for i in range(N):
        vec = test_vectors[i % len(test_vectors)]
        norm = ((np.array([vec], dtype=np.float32) - mean) / scale).astype(np.float32)
        t0 = time.perf_counter_ns()
        sess_quant.run(None, {"microgrid_telemetry": norm})
        t1 = time.perf_counter_ns()
        latencies_quant.append((t1 - t0) / 1000.0)
    
    lat_orig = np.array(latencies_orig)
    lat_quant = np.array(latencies_quant)
    
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK: FP32 vs INT8")
    print("=" * 60)
    print(f"{'Metric':<20} | {'FP32':<12} | {'INT8':<12} | {'Speedup'}")
    print("-" * 55)
    print(f"{'Mean (µs)':<20} | {np.mean(lat_orig):<12.2f} | {np.mean(lat_quant):<12.2f} | {np.mean(lat_orig)/np.mean(lat_quant):.2f}x")
    print(f"{'Median (µs)':<20} | {np.median(lat_orig):<12.2f} | {np.median(lat_quant):<12.2f} | {np.median(lat_orig)/np.median(lat_quant):.2f}x")
    print(f"{'p95 (µs)':<20} | {np.percentile(lat_orig, 95):<12.2f} | {np.percentile(lat_quant, 95):<12.2f} | {np.percentile(lat_orig, 95)/np.percentile(lat_quant, 95):.2f}x")
    print(f"{'p99 (µs)':<20} | {np.percentile(lat_orig, 99):<12.2f} | {np.percentile(lat_quant, 99):<12.2f} | {np.percentile(lat_orig, 99)/np.percentile(lat_quant, 99):.2f}x")
    print(f"{'Throughput (inf/s)':<20} | {1_000_000/np.mean(lat_orig):<12,.0f} | {1_000_000/np.mean(lat_quant):<12,.0f} | {np.mean(lat_orig)/np.mean(lat_quant):.2f}x")
    
    return lat_orig, lat_quant


if __name__ == "__main__":
    quantize_model()
    benchmark_models()
//! Linux hardware actuation module for TEJAS edge agent.
//!
//! This module handles physical actuation via Linux sysfs (`/sys/class`).
//! It provides an async function to execute hardware actions (cooling, relay, bypass)
//! with appropriate PWM and GPIO control.
//!
//! On non-Linux systems, the module will print a simulated success message
//! to prevent crashes during testing or development on other platforms.

use anyhow::{Result, Context};
use std::path::Path;
use tokio::fs;

/// PWM chip path (common on Raspberry Pi, BeagleBone, and other ARM SBCs)
const PWM_CHIP_PATH: &str = "/sys/class/pwm/pwmchip0";
const PWM_CHANNEL_PATH: &str = "pwm0";

/// GPIO pin for relay control
const RELAY_GPIO_PIN: &str = "17";
const RELAY_VALUE_PATH: &str = "/sys/class/gpio";

/// Execute a hardware action based on the inference result.
///
/// # Arguments
/// * `action` - The action to execute ("cooling", "relay", or "bypass")
/// * `throttle` - The throttle level (0.0 to 1.0), interpreted as PWM duty cycle
///
/// # Safety
/// - Checks if running on Linux via `/sys/class` existence
/// - Returns simulated success on non-Linux platforms to prevent crashes
///
/// # Hardware Actions
/// - "cooling": Sets PWM duty cycle scaled from throttle (0.0-1.0 → 0-255)
/// - "relay": Sets GPIO 17 high (value "1")
/// - "bypass": Sets GPIO 17 low (value "0") and PWM duty cycle to 0
pub async fn execute_hardware_action(action: &str, throttle: f32) -> Result<()> {
    // Check if running on Linux by testing for /sys/class existence
    let is_linux = Path::new("/sys/class").exists();

    if !is_linux {
        // Simulation mode for non-Linux environments
        println!(
            "[SIMULATION] Would execute action='{}', throttle={:.3} on Linux hardware",
            action, throttle
        );
        println!(
            "[SIMULATION] PWM: {}, GPIO {}: N/A (simulation mode)",
            if action == "cooling" {
                format!("duty_cycle={}", pwm_throttle(throttle))
            } else {
                "N/A".to_string()
            },
            RELAY_GPIO_PIN
        );
        return Ok(());
    }

    // Real hardware execution on Linux
    match action {
        "cooling" => execute_cooling(throttle).await,
        "relay" => execute_relay().await,
        "bypass" => execute_bypass().await,
        other => Err(anyhow::anyhow!("Unknown action: {}", other)),
    }
}

/// Scale throttle from 0.0-1.0 to PWM duty cycle 0-255
fn pwm_throttle(throttle: f32) -> u16 {
    (throttle.clamp(0.0, 1.0) * 255.0).round() as u16
}

/// Execute cooling action: set PWM duty cycle based on throttle level
async fn execute_cooling(throttle: f32) -> Result<()> {
    let pwm_duty = pwm_throttle(throttle);

    // Export PWM channel if needed (write "0" to export creates the channel)
    let export_path = format!("{}/export", PWM_CHIP_PATH);
    if Path::new(&export_path).exists() {
        let _ = fs::write(&export_path, "0").await;
    }

    // Enable PWM
    let enable_path = format!("{}/{}/enable", PWM_CHIP_PATH, PWM_CHANNEL_PATH);
    let _ = fs::write(&enable_path, "1").await;

    // Set duty cycle
    let duty_path = format!("{}/{}/duty_cycle", PWM_CHIP_PATH, PWM_CHANNEL_PATH);
    fs::write(&duty_path, pwm_duty.to_string())
        .await
        .with_context(|| format!("Failed to write PWM duty_cycle to {}", duty_path))?;

    println!(
        "[HARDWARE] Cooling: PWM duty_cycle={}",
        pwm_duty
    );

    Ok(())
}

/// Execute relay action: set GPIO 17 high
async fn execute_relay() -> Result<()> {
    // Export GPIO 17 if not already exported
    let export_path = format!("{}/export", RELAY_VALUE_PATH);
    let gpio_export_path = format!("{}/gpio{}", RELAY_VALUE_PATH, RELAY_GPIO_PIN);

    if Path::new(&gpio_export_path).exists() {
        let _ = fs::write(&export_path, RELAY_GPIO_PIN).await;
    }

    // Set GPIO direction to output
    let dir_path = format!("{}/gpio{}/direction", RELAY_VALUE_PATH, RELAY_GPIO_PIN);
    fs::write(&dir_path, "out")
        .await
        .with_context(|| format!("Failed to set GPIO {} direction", RELAY_GPIO_PIN))?;

    // Set GPIO value to high (1)
    let value_path = format!("{}/gpio{}/value", RELAY_VALUE_PATH, RELAY_GPIO_PIN);
    fs::write(&value_path, "1")
        .await
        .with_context(|| format!("Failed to set GPIO {} value to 1", RELAY_GPIO_PIN))?;

    println!("[HARDWARE] Relay: GPIO 17 set HIGH");

    Ok(())
}

/// Execute bypass action: set GPIO 17 low and PWM to 0
async fn execute_bypass() -> Result<()> {
    // Disable PWM (set duty cycle to 0)
    let duty_path = format!("{}/{}/duty_cycle", PWM_CHIP_PATH, PWM_CHANNEL_PATH);
    let _ = fs::write(&duty_path, "0").await;

    // Disable PWM
    let enable_path = format!("{}/{}/enable", PWM_CHIP_PATH, PWM_CHANNEL_PATH);
    let _ = fs::write(&enable_path, "0").await;

    // Set GPIO 17 low (0)
    let export_path = format!("{}/export", RELAY_VALUE_PATH);
    let gpio_export_path = format!("{}/gpio{}", RELAY_VALUE_PATH, RELAY_GPIO_PIN);

    if Path::new(&gpio_export_path).exists() {
        let _ = fs::write(&export_path, RELAY_GPIO_PIN).await;
    }

    // Set GPIO direction to output
    let dir_path = format!("{}/gpio{}/direction", RELAY_VALUE_PATH, RELAY_GPIO_PIN);
    fs::write(&dir_path, "out")
        .await
        .with_context(|| format!("Failed to set GPIO {} direction", RELAY_GPIO_PIN))?;

    // Set GPIO value to low (0)
    let value_path = format!("{}/gpio{}/value", RELAY_VALUE_PATH, RELAY_GPIO_PIN);
    fs::write(&value_path, "0")
        .await
        .with_context(|| format!("Failed to set GPIO {} value to 0", RELAY_GPIO_PIN))?;

    println!("[HARDWARE] Bypass: GPIO 17 set LOW, PWM disabled");

    Ok(())
}
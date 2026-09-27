//! Runtime backend selection.
//!
//! The binary is compiled with every backend we can ship on Linux (CUDA, ROCm, wgpu and
//! ndarray). At startup we look at which GPU vendors the kernel reports, then *probe* each
//! candidate backend by running a tiny kernel inside `catch_unwind`: a backend is only picked
//! if its driver libraries actually load and execute. Order: CUDA → ROCm → wgpu → CPU.
//!
//! `NMT_BACKEND=cuda|rocm|wgpu|cpu` forces a backend and skips detection.

use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::{ElementConversion, Tensor};
use burn::tensor::backend::Backend;

#[cfg(target_os = "linux")]
use burn::backend::{Cuda, Rocm, cuda::CudaDevice, rocm::RocmDevice};

#[derive(Debug, Clone)]
pub enum Selected {
    #[cfg(target_os = "linux")]
    Cuda(CudaDevice),
    #[cfg(target_os = "linux")]
    Rocm(RocmDevice),
    Wgpu(WgpuDevice),
    Cpu,
}

impl fmt::Display for Selected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(target_os = "linux")]
            Selected::Cuda(d) => write!(f, "CUDA ({d:?})"),
            #[cfg(target_os = "linux")]
            Selected::Rocm(d) => write!(f, "ROCm ({d:?})"),
            Selected::Wgpu(d) => write!(f, "wgpu/Vulkan ({d:?})"),
            Selected::Cpu => write!(f, "CPU (ndarray)"),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Vendors {
    nvidia: bool,
    amd: bool,
    intel: bool,
}

impl Vendors {
    fn any(self) -> bool {
        self.nvidia || self.amd || self.intel
    }
}

/// Reads PCI vendor IDs of DRM devices (`/sys/class/drm/card*/device/vendor`).
fn detect_vendors() -> Vendors {
    let mut v = Vendors::default();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return v;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // `card0`, `card1`, ... but not connectors like `card0-HDMI-A-1`.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let Ok(id) = std::fs::read_to_string(entry.path().join("device/vendor")) else {
            continue;
        };
        match id.trim() {
            "0x10de" => v.nvidia = true,
            "0x1002" => v.amd = true,
            "0x8086" => v.intel = true,
            _ => {}
        }
    }
    // The proprietary NVIDIA driver does not always expose a DRM card node.
    if std::path::Path::new("/proc/driver/nvidia/version").exists() {
        v.nvidia = true;
    }
    v
}

/// Runs `f`, turning a panic into `Err` with the messages of every panic raised meanwhile.
///
/// cubecl drives each device from its own thread: the root cause usually panics *there*
/// and the calling thread only sees a closed channel (`CallError`). The temporary hook
/// therefore records panics from every thread instead of printing them.
fn capture_panics<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    let captured = Arc::new(Mutex::new(Vec::<String>::new()));
    let hook = panic::take_hook();
    {
        let captured = Arc::clone(&captured);
        panic::set_hook(Box::new(move |info| {
            let thread = std::thread::current();
            let name = thread.name().unwrap_or("<unnamed>");
            if let Ok(mut c) = captured.lock() {
                c.push(format!("[thread '{name}'] {info}"));
            }
        }));
    }
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(hook);
    result.map_err(|_| {
        let captured = captured.lock().map(|c| c.join("
")).unwrap_or_default();
        if captured.is_empty() { "unknown panic".to_string() } else { captured }
    })
}

/// Runs a tiny computation on `device`. Fails with the panic messages if the backend is
/// unusable (missing driver, missing runtime library, no adapter, ...).
fn probe<B: Backend>(device: &B::Device) -> Result<(), String> {
    let sum = capture_panics(|| {
        let t = Tensor::<B, 1>::ones([8], device);
        t.sum().into_scalar().elem::<f32>()
    })?;
    if (sum - 8.0).abs() < 1e-3 {
        Ok(())
    } else {
        Err(format!("probe computed a wrong result ({sum} instead of 8)"))
    }
}

fn try_cuda() -> Result<Selected, String> {
    #[cfg(target_os = "linux")]
    {
        let device = CudaDevice::default();
        probe::<Cuda<f32, i32>>(&device).map(|()| Selected::Cuda(device))
    }
    #[cfg(not(target_os = "linux"))]
    Err("CUDA backend is only built for Linux".to_string())
}

fn try_rocm() -> Result<Selected, String> {
    #[cfg(target_os = "linux")]
    {
        let device = RocmDevice::default();
        probe::<Rocm<f32, i32>>(&device).map(|()| Selected::Rocm(device))
    }
    #[cfg(not(target_os = "linux"))]
    Err("ROCm backend is only built for Linux".to_string())
}

fn try_wgpu() -> Result<Selected, String> {
    // Only real GPUs: a software (CPU) adapter would be slower than ndarray.
    let mut last_err = String::new();
    for device in [WgpuDevice::DiscreteGpu(0), WgpuDevice::IntegratedGpu(0)] {
        match probe::<Wgpu<f32, i32>>(&device) {
            Ok(()) => return Ok(Selected::Wgpu(device)),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

pub fn select() -> Selected {
    if let Ok(forced) = std::env::var("NMT_BACKEND") {
        let forced = forced.to_lowercase();
        let chosen = match forced.as_str() {
            "cuda" => try_cuda(),
            "rocm" | "hip" => try_rocm(),
            "wgpu" | "vulkan" => try_wgpu(),
            "cpu" | "ndarray" => Ok(Selected::Cpu),
            other => panic!("NMT_BACKEND={other} is invalid (use cuda|rocm|wgpu|cpu)"),
        };
        return chosen
            .unwrap_or_else(|e| panic!("NMT_BACKEND={forced} requested but unavailable:
{e}"));
    }

    let vendors = detect_vendors();
    println!("Detected GPU vendors: {vendors:?}");

    if vendors.nvidia {
        match try_cuda() {
            Ok(s) => return s,
            Err(e) => println!("NVIDIA GPU found but CUDA is unavailable:
{e}"),
        }
    }
    if vendors.amd {
        match try_rocm() {
            Ok(s) => return s,
            Err(e) => println!("AMD GPU found but ROCm is unavailable:
{e}"),
        }
    }
    // wgpu is also tried with no detected vendor: sysfs may be hidden (containers).
    match try_wgpu() {
        Ok(s) => return s,
        Err(e) if vendors.any() => println!("GPU found but wgpu/Vulkan is unavailable:
{e}"),
        Err(_) => {}
    }
    Selected::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_panics_reports_worker_thread_cause() {
        let err = capture_panics(|| {
            let worker = std::thread::Builder::new()
                .name("device".into())
                .spawn(|| panic!("libnvrtc not found"))
                .unwrap();
            worker.join().map_err(|_| "CallError").unwrap()
        })
        .unwrap_err();
        assert!(err.contains("[thread 'device']"), "{err}");
        assert!(err.contains("libnvrtc not found"), "{err}");
        assert!(err.contains("CallError"), "{err}");
    }

    #[test]
    fn capture_panics_passes_through_value() {
        assert_eq!(capture_panics(|| 42), Ok(42));
    }
}

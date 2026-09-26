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

/// Runs a tiny computation on `device`; `false` if the backend panics (missing driver,
/// missing runtime library, no adapter, ...). The panic message is suppressed.
fn probe<B: Backend>(device: &B::Device) -> bool {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let ok = panic::catch_unwind(AssertUnwindSafe(|| {
        let t = Tensor::<B, 1>::ones([8], device);
        let sum: f32 = t.sum().into_scalar().elem();
        (sum - 8.0).abs() < 1e-3
    }))
    .unwrap_or(false);
    panic::set_hook(hook);
    ok
}

fn try_cuda() -> Option<Selected> {
    #[cfg(target_os = "linux")]
    {
        let device = CudaDevice::default();
        if probe::<Cuda<f32, i32>>(&device) {
            return Some(Selected::Cuda(device));
        }
    }
    None
}

fn try_rocm() -> Option<Selected> {
    #[cfg(target_os = "linux")]
    {
        let device = RocmDevice::default();
        if probe::<Rocm<f32, i32>>(&device) {
            return Some(Selected::Rocm(device));
        }
    }
    None
}

fn try_wgpu() -> Option<Selected> {
    // Only real GPUs: a software (CPU) adapter would be slower than ndarray.
    [WgpuDevice::DiscreteGpu(0), WgpuDevice::IntegratedGpu(0)]
        .into_iter()
        .find(|d| probe::<Wgpu<f32, i32>>(d))
        .map(Selected::Wgpu)
}

pub fn select() -> Selected {
    if let Ok(forced) = std::env::var("NMT_BACKEND") {
        let forced = forced.to_lowercase();
        let chosen = match forced.as_str() {
            "cuda" => try_cuda(),
            "rocm" | "hip" => try_rocm(),
            "wgpu" | "vulkan" => try_wgpu(),
            "cpu" | "ndarray" => Some(Selected::Cpu),
            other => panic!("NMT_BACKEND={other} is invalid (use cuda|rocm|wgpu|cpu)"),
        };
        return chosen.unwrap_or_else(|| panic!("NMT_BACKEND={forced} requested but unavailable"));
    }

    let vendors = detect_vendors();
    println!("Detected GPU vendors: {vendors:?}");

    if vendors.nvidia {
        if let Some(s) = try_cuda() {
            return s;
        }
        println!("NVIDIA GPU found but CUDA is unavailable (driver/libnvrtc missing?)");
    }
    if vendors.amd {
        if let Some(s) = try_rocm() {
            return s;
        }
        println!("AMD GPU found but ROCm is unavailable (libamdhip64 missing?)");
    }
    // wgpu is also tried with no detected vendor: sysfs may be hidden (containers).
    if let Some(s) = try_wgpu() {
        return s;
    }
    if vendors.any() {
        println!("GPU found but no usable backend (Vulkan driver missing?)");
    }
    Selected::Cpu
}

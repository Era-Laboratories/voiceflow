//! C FFI bindings for VoiceFlow - for Swift/macOS app integration
//!
//! Build: cargo build --release -p voiceflow-ffi
//! This generates a dylib/staticlib that can be linked from Swift

use std::ffi::{c_char, c_float, CStr, CString};
use std::ptr;
use std::io::Write;

use voiceflow_core::{Config, Pipeline};

/// Write debug log to file (since macOS GUI apps don't have stderr)
fn log_debug(msg: &str) {
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/voiceflow_debug.log")
    {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(file, "[{}] {}", timestamp, msg);
    }
}

/// Opaque handle to the VoiceFlow pipeline
pub struct VoiceFlowHandle {
    pipeline: Pipeline,
}

/// Result struct returned to foreign callers
#[repr(C)]
pub struct VoiceFlowResult {
    pub success: bool,
    pub formatted_text: *mut c_char,
    pub raw_transcript: *mut c_char,
    pub error_message: *mut c_char,
    pub transcription_ms: u64,
    pub llm_ms: u64,
    pub total_ms: u64,
}

/// Initialize the VoiceFlow pipeline
///
/// # Safety
/// config_path must be a valid null-terminated string or null for default
#[no_mangle]
pub unsafe extern "C" fn voiceflow_init(config_path: *const c_char) -> *mut VoiceFlowHandle {
    log_debug("voiceflow_init called");

    // Wrap everything in catch_unwind to prevent panics from unwinding across FFI boundary
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let config_str = if config_path.is_null() {
            None
        } else {
            match CStr::from_ptr(config_path).to_str() {
                Ok(s) => Some(s),
                Err(_) => return ptr::null_mut(),
            }
        };

        let config = match Config::load(config_str) {
            Ok(c) => {
                log_debug(&format!("Config loaded: STT={:?}", c.stt_engine));
                c
            },
            Err(e) => {
                log_debug(&format!("Failed to load config: {}", e));
                return ptr::null_mut();
            }
        };

        log_debug("Creating pipeline (loading ONNX models - this may take a while)...");
        let pipeline = match Pipeline::new(&config) {
            Ok(p) => {
                log_debug("Pipeline created successfully");
                p
            },
            Err(e) => {
                log_debug(&format!("Failed to create pipeline: {}", e));
                return ptr::null_mut();
            }
        };

        log_debug("voiceflow_init complete - returning handle");
        Box::into_raw(Box::new(VoiceFlowHandle { pipeline }))
    }));

    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            log_debug(&format!("PANIC caught in voiceflow_init: {}", msg));
            ptr::null_mut()
        }
    }
}

/// Process audio samples and return formatted text
///
/// # Safety
/// - handle must be a valid pointer from voiceflow_init
/// - audio_data must point to audio_len floats (16kHz mono PCM)
/// - context can be null
#[no_mangle]
pub unsafe extern "C" fn voiceflow_process(
    handle: *mut VoiceFlowHandle,
    audio_data: *const c_float,
    audio_len: usize,
    context: *const c_char,
) -> VoiceFlowResult {
    log_debug(&format!("voiceflow_process called with {} samples", audio_len));

    if handle.is_null() || audio_data.is_null() {
        log_debug("ERROR - Invalid handle or audio data");
        return error_result("Invalid handle or audio data");
    }

    // Store raw pointers for use in closure
    let handle_ptr = handle;
    let audio_ptr = audio_data;
    let context_ptr = context;

    // Wrap in catch_unwind to prevent panics from unwinding across FFI boundary
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let handle = &mut *handle_ptr;
        let audio = std::slice::from_raw_parts(audio_ptr, audio_len);

        // Log audio stats
        let audio_duration = audio_len as f32 / 16000.0;
        let max_val = audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        log_debug(&format!("Audio duration: {:.2}s, max amplitude: {:.4}", audio_duration, max_val));

        let context_str = if context_ptr.is_null() {
            None
        } else {
            match CStr::from_ptr(context_ptr).to_str() {
                Ok(s) => Some(s),
                Err(_) => None,
            }
        };

        log_debug("Calling pipeline.process()...");
        match handle.pipeline.process(audio, context_str) {
            Ok(result) => {
                log_debug(&format!("Success! Raw transcript: '{}'", result.raw_transcript));
                log_debug(&format!("Formatted text: '{}'", result.formatted_text));
                VoiceFlowResult {
                    success: true,
                    formatted_text: CString::new(result.formatted_text)
                        .map(|s| s.into_raw())
                        .unwrap_or(ptr::null_mut()),
                    raw_transcript: CString::new(result.raw_transcript)
                        .map(|s| s.into_raw())
                        .unwrap_or(ptr::null_mut()),
                    error_message: ptr::null_mut(),
                    transcription_ms: result.timings.transcription_ms,
                    llm_ms: result.timings.llm_formatting_ms,
                    total_ms: result.timings.total_ms,
                }
            },
            Err(e) => {
                log_debug(&format!("ERROR - pipeline.process failed: {}", e));
                error_result(&e.to_string())
            },
        }
    }));

    match result {
        Ok(vf_result) => vf_result,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            log_debug(&format!("PANIC caught in voiceflow_process: {}", msg));
            error_result(&format!("Internal error: {}", msg))
        }
    }
}

/// Free a VoiceFlowResult's strings
///
/// # Safety
/// Only call this once per result
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_result(result: VoiceFlowResult) {
    if !result.formatted_text.is_null() {
        let _ = CString::from_raw(result.formatted_text);
    }
    if !result.raw_transcript.is_null() {
        let _ = CString::from_raw(result.raw_transcript);
    }
    if !result.error_message.is_null() {
        let _ = CString::from_raw(result.error_message);
    }
}

/// Cleanup and free the handle
///
/// # Safety
/// Only call this once per handle
#[no_mangle]
pub unsafe extern "C" fn voiceflow_destroy(handle: *mut VoiceFlowHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Get the library version
#[no_mangle]
pub extern "C" fn voiceflow_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

fn error_result(msg: &str) -> VoiceFlowResult {
    VoiceFlowResult {
        success: false,
        formatted_text: ptr::null_mut(),
        raw_transcript: ptr::null_mut(),
        error_message: CString::new(msg)
            .map(|s| s.into_raw())
            .unwrap_or(ptr::null_mut()),
        transcription_ms: 0,
        llm_ms: 0,
        total_ms: 0,
    }
}

/// Model info struct for FFI
#[repr(C)]
pub struct ModelInfo {
    pub id: *mut c_char,
    pub display_name: *mut c_char,
    pub filename: *mut c_char,
    pub size_gb: c_float,
    pub is_downloaded: bool,
}

/// Get the models directory path
#[no_mangle]
pub extern "C" fn voiceflow_models_dir() -> *mut c_char {
    match Config::models_dir() {
        Ok(path) => CString::new(path.to_string_lossy().to_string())
            .map(|s| s.into_raw())
            .unwrap_or(ptr::null_mut()),
        Err(_) => ptr::null_mut(),
    }
}

/// Get the number of available models
#[no_mangle]
pub extern "C" fn voiceflow_model_count() -> usize {
    use voiceflow_core::config::LlmModel;
    LlmModel::all_models().len()
}

/// Get model info by index
///
/// # Safety
/// index must be < voiceflow_model_count()
#[no_mangle]
pub unsafe extern "C" fn voiceflow_model_info(index: usize) -> ModelInfo {
    use voiceflow_core::config::LlmModel;

    let models = LlmModel::all_models();
    if index >= models.len() {
        return ModelInfo {
            id: ptr::null_mut(),
            display_name: ptr::null_mut(),
            filename: ptr::null_mut(),
            size_gb: 0.0,
            is_downloaded: false,
        };
    }

    let model = &models[index];
    let models_dir = Config::models_dir().ok();
    let is_downloaded = models_dir
        .map(|dir| dir.join(model.filename()).exists())
        .unwrap_or(false);

    let id_str = match model {
        LlmModel::Qwen3_1_7B => "qwen3-1.7b",
        LlmModel::Qwen3_4B => "qwen3-4b",
        LlmModel::SmolLM3_3B => "smollm3-3b",
        LlmModel::Gemma2_2B => "gemma2-2b",
        LlmModel::Phi2 => "phi-2",
        LlmModel::Custom(_) => "custom",
    };

    ModelInfo {
        id: CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        display_name: CString::new(model.display_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        filename: CString::new(model.filename()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        size_gb: model.size_gb(),
        is_downloaded,
    }
}

/// Free model info strings
///
/// # Safety
/// Only call once per ModelInfo
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_model_info(info: ModelInfo) {
    if !info.id.is_null() {
        let _ = CString::from_raw(info.id);
    }
    if !info.display_name.is_null() {
        let _ = CString::from_raw(info.display_name);
    }
    if !info.filename.is_null() {
        let _ = CString::from_raw(info.filename);
    }
}

/// Free a C string returned by other functions
///
/// # Safety
/// Only call once per string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Get the current model ID from config
#[no_mangle]
pub extern "C" fn voiceflow_current_model() -> *mut c_char {
    use voiceflow_core::config::LlmModel;

    let config = Config::load(None).unwrap_or_default();
    let id_str = match config.llm_model {
        LlmModel::Qwen3_1_7B => "qwen3-1.7b",
        LlmModel::Qwen3_4B => "qwen3-4b",
        LlmModel::SmolLM3_3B => "smollm3-3b",
        LlmModel::Gemma2_2B => "gemma2-2b",
        LlmModel::Phi2 => "phi-2",
        LlmModel::Custom(_) => "custom",
    };

    CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the current model in config (requires restart to take effect)
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_model(model_id: *const c_char) -> bool {
    use voiceflow_core::config::LlmModel;

    if model_id.is_null() {
        return false;
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match id_str {
        "qwen3-1.7b" => LlmModel::Qwen3_1_7B,
        "qwen3-4b" => LlmModel::Qwen3_4B,
        "smollm3-3b" => LlmModel::SmolLM3_3B,
        "gemma2-2b" => LlmModel::Gemma2_2B,
        "phi-2" => LlmModel::Phi2,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.llm_model = model;
    config.save(None).is_ok()
}

/// Get the HuggingFace download URL for a model
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_model_download_url(model_id: *const c_char) -> *mut c_char {
    use voiceflow_core::config::LlmModel;

    if model_id.is_null() {
        return ptr::null_mut();
    }

    let id_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let model = match id_str {
        "qwen3-1.7b" => LlmModel::Qwen3_1_7B,
        "qwen3-4b" => LlmModel::Qwen3_4B,
        "smollm3-3b" => LlmModel::SmolLM3_3B,
        "gemma2-2b" => LlmModel::Gemma2_2B,
        "phi-2" => LlmModel::Phi2,
        _ => return ptr::null_mut(),
    };

    if let Some(repo) = model.hf_repo() {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo,
            model.filename()
        );
        CString::new(url).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
    } else {
        ptr::null_mut()
    }
}

// =============================================================================
// STT Engine Management
// =============================================================================

/// Get the current STT engine ("whisper" or "moonshine")
#[no_mangle]
pub extern "C" fn voiceflow_current_stt_engine() -> *mut c_char {
    use voiceflow_core::config::SttEngine;

    let config = Config::load(None).unwrap_or_default();
    let engine_str = match config.stt_engine {
        SttEngine::Whisper => "whisper",
        SttEngine::Moonshine => "moonshine",
    };

    CString::new(engine_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the current STT engine ("whisper" or "moonshine")
///
/// # Safety
/// engine_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_stt_engine(engine_id: *const c_char) -> bool {
    use voiceflow_core::config::SttEngine;

    if engine_id.is_null() {
        return false;
    }

    let engine_str = match CStr::from_ptr(engine_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let engine = match engine_str {
        "whisper" => SttEngine::Whisper,
        "moonshine" => SttEngine::Moonshine,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.stt_engine = engine;
    config.save(None).is_ok()
}

/// Get the current Moonshine model ("tiny" or "base")
#[no_mangle]
pub extern "C" fn voiceflow_current_moonshine_model() -> *mut c_char {
    use voiceflow_core::config::MoonshineModel;

    let config = Config::load(None).unwrap_or_default();
    let model_str = match config.moonshine_model {
        MoonshineModel::Tiny => "tiny",
        MoonshineModel::Base => "base",
    };

    CString::new(model_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut())
}

/// Set the current Moonshine model ("tiny" or "base")
///
/// # Safety
/// model_id must be a valid null-terminated string
#[no_mangle]
pub unsafe extern "C" fn voiceflow_set_moonshine_model(model_id: *const c_char) -> bool {
    use voiceflow_core::config::MoonshineModel;

    if model_id.is_null() {
        return false;
    }

    let model_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match model_str {
        "tiny" => MoonshineModel::Tiny,
        "base" => MoonshineModel::Base,
        _ => return false,
    };

    let mut config = Config::load(None).unwrap_or_default();
    config.moonshine_model = model;
    config.save(None).is_ok()
}

/// Moonshine model info struct for FFI
#[repr(C)]
pub struct MoonshineModelInfo {
    pub id: *mut c_char,
    pub display_name: *mut c_char,
    pub size_mb: u32,
    pub is_downloaded: bool,
}

/// Get the number of available Moonshine models
#[no_mangle]
pub extern "C" fn voiceflow_moonshine_model_count() -> usize {
    2 // Tiny and Base
}

/// Get Moonshine model info by index
///
/// # Safety
/// index must be < voiceflow_moonshine_model_count()
#[no_mangle]
pub unsafe extern "C" fn voiceflow_moonshine_model_info(index: usize) -> MoonshineModelInfo {
    use voiceflow_core::config::MoonshineModel;

    let model = match index {
        0 => MoonshineModel::Tiny,
        1 => MoonshineModel::Base,
        _ => return MoonshineModelInfo {
            id: ptr::null_mut(),
            display_name: ptr::null_mut(),
            size_mb: 0,
            is_downloaded: false,
        },
    };

    let config = Config::load(None).unwrap_or_default();
    let is_downloaded = config.moonshine_model_downloaded_for(&model);

    let id_str = match model {
        MoonshineModel::Tiny => "tiny",
        MoonshineModel::Base => "base",
    };

    MoonshineModelInfo {
        id: CString::new(id_str).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        display_name: CString::new(model.display_name()).map(|s| s.into_raw()).unwrap_or(ptr::null_mut()),
        size_mb: model.size_mb(),
        is_downloaded,
    }
}

/// Free Moonshine model info strings
///
/// # Safety
/// Only call once per MoonshineModelInfo
#[no_mangle]
pub unsafe extern "C" fn voiceflow_free_moonshine_model_info(info: MoonshineModelInfo) {
    if !info.id.is_null() {
        let _ = CString::from_raw(info.id);
    }
    if !info.display_name.is_null() {
        let _ = CString::from_raw(info.display_name);
    }
}

/// Check if a Moonshine model is downloaded
///
/// # Safety
/// model_id must be a valid null-terminated string ("tiny" or "base")
#[no_mangle]
pub unsafe extern "C" fn voiceflow_moonshine_model_downloaded(model_id: *const c_char) -> bool {
    use voiceflow_core::config::MoonshineModel;

    if model_id.is_null() {
        return false;
    }

    let model_str = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let model = match model_str {
        "tiny" => MoonshineModel::Tiny,
        "base" => MoonshineModel::Base,
        _ => return false,
    };

    let config = Config::load(None).unwrap_or_default();
    config.moonshine_model_downloaded_for(&model)
}

/// Get the Moonshine models directory path
#[no_mangle]
pub extern "C" fn voiceflow_moonshine_models_dir() -> *mut c_char {
    let config = Config::load(None).unwrap_or_default();
    match config.moonshine_model_dir() {
        Ok(path) => {
            // Return parent directory (models dir)
            if let Some(parent) = path.parent() {
                CString::new(parent.to_string_lossy().to_string())
                    .map(|s| s.into_raw())
                    .unwrap_or(ptr::null_mut())
            } else {
                ptr::null_mut()
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

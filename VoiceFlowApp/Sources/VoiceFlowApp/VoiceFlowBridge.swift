import Foundation
import VoiceFlowFFI

/// Swift wrapper for VoiceFlow Rust library
@MainActor
final class VoiceFlowBridge: ObservableObject {
    private var handle: OpaquePointer?
    private var isCleanedUp = false
    private var asrEngine: Qwen3ASREngine?

    @Published var isInitialized = false
    @Published var isProcessing = false
    @Published var lastError: String?
    @Published var lastResult: ProcessResult?
    @Published var isConsolidatedMode = false
    /// Whether the STT engine is external (Qwen3-ASR Python daemon) in traditional mode
    @Published var isExternalStt = false

    struct ProcessResult {
        let formattedText: String
        let rawTranscript: String
        let transcriptionMs: UInt64
        let llmMs: UInt64
        let totalMs: UInt64
    }

    /// Memory usage information from the Rust side
    struct MemoryUsage {
        let residentBytes: UInt
        let virtualBytes: UInt
        let peakBytes: UInt

        var residentMB: Double { Double(residentBytes) / 1024 / 1024 }
        var virtualMB: Double { Double(virtualBytes) / 1024 / 1024 }
        var peakMB: Double { Double(peakBytes) / 1024 / 1024 }
    }

    init() {
        // Initialize on background thread to avoid blocking
        Task.detached { [weak self] in
            await self?.initialize()
        }
    }

    deinit {
        // Perform FFI cleanup directly in deinit (thread-safe)
        // Note: can't update @Published properties here, just release resources
        if let handle = handle {
            voiceflow_unload_models(handle)
            voiceflow_prepare_shutdown()
            voiceflow_destroy(handle)
        }
    }

    /// Explicit cleanup - call before app termination for clean shutdown
    func cleanup() {
        guard !isCleanedUp else { return }
        isCleanedUp = true

        // Stop the Python ASR daemon if running
        asrEngine?.stopDaemon()
        asrEngine = nil

        if let handle = handle {
            // Unload models first to release memory
            voiceflow_unload_models(handle)

            // Prepare for shutdown
            voiceflow_prepare_shutdown()

            // Destroy the handle
            voiceflow_destroy(handle)
            self.handle = nil
        }

        isInitialized = false
    }

    /// Unload models from memory without destroying the handle
    /// Models will be reloaded on next use
    func unloadModels() {
        guard let handle = handle else { return }
        voiceflow_unload_models(handle)
    }

    /// Reset the LLM engine, allowing re-initialization on next use
    func resetLLM() {
        guard let handle = handle else { return }
        voiceflow_reset_llm(handle)
    }

    /// Get current memory usage from the Rust side
    nonisolated static func getMemoryUsage() -> MemoryUsage {
        let info = voiceflow_memory_info()
        return MemoryUsage(
            residentBytes: info.resident_bytes,
            virtualBytes: info.virtual_bytes,
            peakBytes: info.peak_bytes
        )
    }

    /// Force garbage collection hint
    nonisolated static func forceGC() {
        voiceflow_force_gc()
    }

    private func initialize() async {
        let consolidatedMode = voiceflow_is_consolidated_mode()
        let externalStt = voiceflow_is_external_stt()

        // Initialize with default config
        let newHandle = voiceflow_init(nil)

        await MainActor.run {
            self.handle = newHandle
            self.isConsolidatedMode = consolidatedMode
            self.isExternalStt = externalStt
            self.isInitialized = newHandle != nil
            if newHandle == nil {
                self.lastError = "Failed to initialize VoiceFlow pipeline"
            }

            // Start Qwen3-ASR daemon if needed (consolidated mode OR external STT in traditional mode)
            let needsAsrDaemon = consolidatedMode || externalStt
            if needsAsrDaemon {
                self.asrEngine = Qwen3ASREngine()

                // Load model in background
                if let engine = self.asrEngine {
                    Task {
                        // Get current consolidated model ID from config
                        guard let modelIdPtr = voiceflow_current_consolidated_model() else {
                            NSLog("[VoiceFlowBridge] No consolidated model configured")
                            return
                        }
                        let modelId = String(cString: modelIdPtr)
                        voiceflow_free_string(modelIdPtr)

                        // Get the model directory path
                        guard let modelDirPtr = modelId.withCString({ cStr in
                            voiceflow_consolidated_model_dir(cStr)
                        }) else {
                            NSLog("[VoiceFlowBridge] Failed to get model dir for \(modelId)")
                            return
                        }
                        let modelDir = String(cString: modelDirPtr)
                        voiceflow_free_string(modelDirPtr)

                        NSLog("[VoiceFlowBridge] Loading Qwen3-ASR model from: \(modelDir)")
                        await engine.loadModel(from: modelDir)
                    }
                }
            }
        }
    }

    /// Process audio samples (16kHz mono float PCM)
    func process(audio: [Float], context: String? = nil) async -> ProcessResult? {
        guard let handle = handle else {
            await MainActor.run {
                self.lastError = "Pipeline not initialized"
            }
            return nil
        }

        await MainActor.run {
            self.isProcessing = true
            self.lastError = nil
        }

        defer {
            Task { @MainActor in
                self.isProcessing = false
            }
        }

        // Consolidated mode: Qwen3-ASR + Rust post-processing only (no LLM)
        if isConsolidatedMode, let asrEngine = asrEngine {
            let asrStart = CFAbsoluteTimeGetCurrent()
            guard let rawText = await asrEngine.process(audio: audio) else {
                await MainActor.run {
                    self.lastError = "Qwen3-ASR transcription failed"
                }
                return nil
            }
            let asrMs = UInt64((CFAbsoluteTimeGetCurrent() - asrStart) * 1000)

            // Apply Rust post-processing (filler removal, tokenization fix, voice commands, replacements)
            let postProcessed: String = rawText.withCString { cStr in
                guard let resultPtr = voiceflow_post_process_text(cStr) else {
                    return rawText
                }
                let result = String(cString: resultPtr)
                voiceflow_free_string(resultPtr)
                return result
            }

            let processResult = ProcessResult(
                formattedText: postProcessed,
                rawTranscript: rawText,
                transcriptionMs: asrMs,
                llmMs: 0,
                totalMs: asrMs
            )

            await MainActor.run {
                self.lastResult = processResult
            }

            return processResult
        }

        // External STT mode (Qwen3-ASR in traditional pipeline): ASR daemon + LLM formatting
        if isExternalStt, let asrEngine = asrEngine {
            let asrStart = CFAbsoluteTimeGetCurrent()
            guard let rawText = await asrEngine.process(audio: audio) else {
                await MainActor.run {
                    self.lastError = "Qwen3-ASR transcription failed"
                }
                return nil
            }
            let asrMs = UInt64((CFAbsoluteTimeGetCurrent() - asrStart) * 1000)

            // Send raw text through Rust pipeline for post-processing + LLM formatting
            let result: VoiceFlowResult = await withCheckedContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    let ffiResult = rawText.withCString { textPtr in
                        if let ctx = context {
                            return ctx.withCString { ctxPtr in
                                voiceflow_format_text(handle, textPtr, ctxPtr)
                            }
                        } else {
                            return voiceflow_format_text(handle, textPtr, nil)
                        }
                    }
                    continuation.resume(returning: ffiResult)
                }
            }

            defer {
                voiceflow_free_result(result)
            }

            if result.success {
                let processResult = ProcessResult(
                    formattedText: result.formatted_text.map { String(cString: $0) } ?? "",
                    rawTranscript: rawText,
                    transcriptionMs: asrMs,
                    llmMs: result.llm_ms,
                    totalMs: asrMs + result.llm_ms
                )

                await MainActor.run {
                    self.lastResult = processResult
                }

                return processResult
            } else {
                let errorMsg = result.error_message.map { String(cString: $0) } ?? "LLM formatting failed"
                await MainActor.run {
                    self.lastError = errorMsg
                }
                return nil
            }
        }

        // Traditional mode (Whisper/Moonshine): call FFI pipeline on background thread
        let result: VoiceFlowResult = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let ffiResult = audio.withUnsafeBufferPointer { buffer in
                    let count = UInt(buffer.count)
                    if let ctx = context {
                        return ctx.withCString { ctxPtr in
                            voiceflow_process(handle, buffer.baseAddress, count, ctxPtr)
                        }
                    } else {
                        return voiceflow_process(handle, buffer.baseAddress, count, nil)
                    }
                }
                continuation.resume(returning: ffiResult)
            }
        }

        defer {
            voiceflow_free_result(result)
        }

        if result.success {
            let processResult = ProcessResult(
                formattedText: result.formatted_text.map { String(cString: $0) } ?? "",
                rawTranscript: result.raw_transcript.map { String(cString: $0) } ?? "",
                transcriptionMs: result.transcription_ms,
                llmMs: result.llm_ms,
                totalMs: result.total_ms
            )

            await MainActor.run {
                self.lastResult = processResult
            }

            return processResult
        } else {
            let errorMsg = result.error_message.map { String(cString: $0) } ?? "Unknown error"
            await MainActor.run {
                self.lastError = errorMsg
            }
            return nil
        }
    }

    /// Get library version
    nonisolated static var version: String {
        guard let versionPtr = voiceflow_version() else {
            return "unknown"
        }
        return String(cString: versionPtr)
    }
}

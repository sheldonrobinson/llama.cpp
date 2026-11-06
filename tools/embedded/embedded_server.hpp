#pragma once

#ifdef DeleteFile
#undef DeleteFile
#endif

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <future>
#include <chrono>
#include <optional>
#include <nlohmann/json.hpp>

#include "libopenai/libopenai.hpp"
#include "libopenai/transport.hpp"
#include "libopenai/inference_backend.hpp"
#include "libopenai/inference_client.hpp"
#include "libopenai/inference_provider.hpp"
#include "libopenai/provider_registry.hpp"
#include "libopenai/device_cloud_router.hpp"

#include "httplib.h"

#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "server-context.h"
#include "server-task.h"
#include "server-queue.h"

namespace embedded {

// Forward declaration for cross-reference between EmbeddedBackend and EmbeddedServer
class EmbeddedServer;

using json = nlohmann::json;

/**
 * @brief EmbeddedBackend - exposes the in-process llama.cpp engine as a
 *        libopenai InferenceBackend.
 *
 * In single-process mode the engine runs server-context in this process, so
 * there is no subprocess and no IPC. The backend just forwards requests to the
 * server_context-based inference engine.
 */
class EmbeddedBackend : public libopenai::InferenceBackend {
public:
    struct Config {
        std::string model_path;           // Path to GGUF model file
        int n_ctx = 0;                    // Context size (0 = trained context)
        int n_batch = 2048;               // Logical batch size
        int n_ubatch = 512;               // Physical batch size
        int n_gpu_layers = -1;            // GPU layers (-1 = auto)
        int n_threads = -1;               // Threads (-1 = auto)
        bool embedding = false;           // Enable embedding mode
        bool verbose = false;
    };

    EmbeddedBackend(std::string name, Config config);

    ~EmbeddedBackend() override;

    std::future<libopenai::Response> ExecuteAsync(const libopenai::Request& req) override;
    libopenai::Response Execute(const libopenai::Request& req) override;

    std::string Name() const override { return name_; }
    std::string Type() const override { return "embedded-llama"; }

    libopenai::BackendHealth Health() const override;
    libopenai::BackendLoad GetLoad() const override;

    // Request processors (declared here so they can be called from ExecuteAsync)
    bool process_models(libopenai::Response& resp);
    bool process_completion(const libopenai::Request& req, libopenai::Response& resp);
    bool process_chat_completion(const libopenai::Request& req, libopenai::Response& resp);
    bool process_embeddings(const libopenai::Request& req, libopenai::Response& resp);

    bool SupportsModel(const std::string& model) const override;

    bool is_model_loaded() const { return model_loaded_; }
    const std::string& model_path() const { return config_.model_path; }

    void set_model_loaded(bool loaded) { model_loaded_.store(loaded, std::memory_order_release); }
    void set_health(libopenai::BackendHealth h) { health_ = h; }

    // Bind the in-process server. Provides access to server_context for
    // single-process (in-memory) inference. Non-owning.
    void set_server(EmbeddedServer* server) { server_ = server; }

protected:
    // Accessible by EmbeddedServer to check if backend is bound to a server
    EmbeddedServer* server() const { return server_; }

private:
    std::string name_;
    Config config_;

    EmbeddedServer* server_ = nullptr; // non-owning, single-process mode only

    std::atomic<bool> model_loaded_{false};
    mutable libopenai::BackendLoad load_;
    mutable std::mutex load_mutex_;
    mutable libopenai::BackendHealth health_ = libopenai::BackendHealth::Unknown;

    void record_request_start();
};

/**
 * @brief EmbeddedServer - serves the llama-server API surface for one model.
 *
 * Uses server_context (from the server-context static library) as the
 * high-level inference integration layer. This provides:
 *  - Automatic speculative decoding via common_params.speculative
 *  - All llama-server endpoints via server_routes
 *  - Thread-safe task queue via server_queue/server_response
 *
 * Two modes:
 *   - single_process = true  : the model is served in this process by the
 *                              server-context library. No subprocess.
 *   - single_process = false : a child llama-server process serves the model
 *                              and this server proxies to it (named pipe mode).
 */
class EmbeddedServer {
public:
    struct Config {
        std::string model_path;          // GGUF model path
        std::string mmproj_path;         // multimodal projector (mmproj) for audio/images, optional
        std::string draft_model_path;    // draft GGUF model path for speculative decoding
        int n_draft = 3;                 // max draft tokens for speculative decoding
        std::string llama_server_bin;    // child binary, multi-process mode only
        std::string pipe_name;           // named pipe name, multi-process mode only
        std::string host = "127.0.0.1";  // HTTP bind host
        int port = 0;                    // HTTP port (0 = ephemeral)
        bool single_process = true;      // true = in-process, false = subprocess
        int n_ctx = 0;
        int n_batch = 2048;
        int n_ubatch = 512;
        int n_gpu_layers = -1;
        int n_threads = -1;
        bool embedding = false;
        std::string api_key;             // optional API key for auth
        int timeout_seconds = 60;

        // --- multimodal projector (mtmd) -------------------------------------
        // Setting mmproj_path makes server_context load the multimodal
        // projector through mtmd. All multimodal endpoints (chat/completions
        // with image/audio parts, /v1/audio/transcriptions, /tts, /v1/realtime)
        // are then served through server_context slots.
        bool mmproj_use_gpu = true;      // offload the mtmd projector to the GPU
        std::string mmproj_device;       // ggml device name for mtmd ("" = default)
        bool no_mmproj = false;          // explicitly disable the projector

        // --- server_context options -----------------------------------------
        int n_parallel = 1;              // number of slots (parallel sequences)
        bool kv_unified = false;         // unified KV cache across the slots
        bool cont_batching = true;       // continuous batching
        bool jinja = true;               // use jinja chat templates
        std::string chat_template;       // optional chat template override
        std::string model_alias;         // optional model name/alias
        bool endpoint_slots = true;      // serve /slots
        bool endpoint_props = true;      // serve GET/POST /props
        bool endpoint_metrics = true;    // serve /metrics
    };

    EmbeddedServer(Config config);
    ~EmbeddedServer();

    bool start();
    void stop();

    bool is_running() const { return running_; }

    std::shared_ptr<libopenai::InferenceClient> create_client();

    std::shared_ptr<libopenai::ProviderRegistry> registry() const { return registry_; }
    std::shared_ptr<libopenai::InferenceClient> client() const { return client_; }

    int port() const { return config_.port; }
    const std::string& host() const { return config_.host; }

    // Execute a request using server_context (single-process mode only)
    libopenai::Response execute_internal(const libopenai::Request& req);

    // Access the underlying server_context (single-process mode only)
    server_context* server_ctx() { return &server_ctx_; }

private:
    Config config_;
    std::atomic<bool> running_{false};

    // libopenai frontend
    std::shared_ptr<libopenai::ProviderRegistry> registry_;
    std::shared_ptr<libopenai::InferenceClient> client_;
    std::shared_ptr<EmbeddedBackend> backend_;

    // single-process serving via server_context (high-level integration).
    // server_routes is constructed from params_/server_ctx_ and implements every
    // llama-server endpoint by submitting real tasks into llama_context via the
    // server_queue. It must be declared after params_ and server_ctx_.
    server_context server_ctx_;
    common_params  params_;

    // high-level llama-server route table (mirrors tools/server/server.cpp
    // server_register_routes). Every handle_* below routes through these.
    std::unique_ptr<server_routes> routes_;
    std::map<std::string, server_http_context::handler_t> route_table_;
    std::atomic<bool> model_ready_{false};

    // Background thread running server_ctx_.start_loop() to process the task queue
    std::thread      server_loop_thread_;

    bool init_server_context();
    void free_model();

    // Populate route_table_ from routes_ (mirrors upstream server_register_routes)
    void register_handlers();
    // Dispatch a request to the matching server_routes handler. Returns nullptr
    // when no handler matches the method+path.
    server_http_res_ptr dispatch_request(const std::string& method,
                                         const std::string& path,
                                         const server_http_req& req);
    // Write a server_http_res into an httplib response (handles streaming)
    void write_http_response(server_http_res_ptr res,
                             std::shared_ptr<std::function<bool()>> should_stop,
                             httplib::Response& out);

    friend class EmbeddedBackend;

    // HTTP server
    std::unique_ptr<httplib::Server> http_server_;
    std::thread http_server_thread_;

    // multimodal projector (mmproj) path for mtmd audio/image support
    std::string mmproj_path_;

    // multimodal context (mtmd) for audio, images, video, realtime
    mtmd_context* mtmd_ctx_ = nullptr;

    // multi-process serving (child llama-server / named-pipe mode)
    bool upstream_on_ = false;
    std::string upstream_url_;
#ifdef _WIN32
    void* llama_server_handle_ = nullptr;
    void* named_pipe_handle_ = nullptr;
#else
    int llama_server_pid_ = 0;
    int named_pipe_fd_ = -1;
#endif
    bool llama_server_fallback_ = false;
    bool named_pipe_running_ = false;
    std::string llama_server_url_;
    std::thread llama_server_monitor_thread_;
    std::thread named_pipe_thread_;

    void register_routes();
    void bind_route(const std::string& method, const std::string& path,
                    const std::string& handler_name);
    void handle_route(const std::string& handler_name,
                      const httplib::Request& req, httplib::Response& res);

    // endpoint handlers - delegate to server_context task submission
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_metrics(const httplib::Request& req, httplib::Response& res);
    void handle_props(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_embeddings(const httplib::Request& req, httplib::Response& res);
    void handle_transcriptions(const httplib::Request& req, httplib::Response& res);
    void handle_tts(const httplib::Request& req, httplib::Response& res);
    void handle_realtime(const httplib::Request& req, httplib::Response& res);
    void handle_anthropic_messages(const httplib::Request& req, httplib::Response& res);
    void handle_infill(const httplib::Request& req, httplib::Response& res);
    void handle_control(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_tokenize(const httplib::Request& req, httplib::Response& res);
    void handle_detokenize(const httplib::Request& req, httplib::Response& res);
    void handle_apply_template(const httplib::Request& req, httplib::Response& res);
    void handle_rerank(const httplib::Request& req, httplib::Response& res);
    void handle_systemone(const httplib::Request& req, httplib::Response& res);
    void handle_token_counting(const httplib::Request& req, httplib::Response& res);
    void handle_lora_adapters(const httplib::Request& req, httplib::Response& res);
    void handle_slots(const httplib::Request& req, httplib::Response& res);
    // multimodal (images, audio, video) via mtmd
    void handle_images(const httplib::Request& req, httplib::Response& res);

    // fall-through proxying
    bool try_libopenai(const httplib::Request& req, httplib::Response& res);
    void proxy_to_llama_server(const httplib::Request& req, httplib::Response& res);
    std::string extract_path(const std::string& path) const;
    bool is_libopenai_endpoint(const std::string& endpoint) const;

    // child process management (multi-process / named-pipe mode only)
    void launch_subprocess();
    void launch_llama_server_fallback();
    void stop_llama_server_fallback();
    void monitor_subprocess();

    // named pipe server (multi-process mode only)
    void start_named_pipe_server();
    void stop_named_pipe_server();
    void named_pipe_listener_thread();
    libopenai::Response handle_named_pipe_request(
        const libopenai::Request& req);

    void build_frontend();

    // Submit a server_task and wait for result (single-process mode)
    libopenai::Response submit_task(const server_task& task,
                                     task_response_type res_type,
                                     const std::string& model_name);

    // Build a server_task from an HTTP request body for completions
    server_task build_completion_task(const json& body,
                                      const std::vector<llama_token>& tokens,
                                      server_task_type type,
                                      int32_t id);

    // Convert server_task_result to libopenai::Response
    libopenai::Response result_to_response(const server_task_result& result,
                                           task_response_type res_type,
                                           const std::string& model_name);
};

} // namespace embedded

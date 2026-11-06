#include "embedded_server.hpp"
#include "speculative.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-models.h"

#ifdef DeleteFile
#undef DeleteFile
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cfloat>
#include <random>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#endif

namespace embedded {

// In-process engine handle, backed by server-context. Defined here so
// llama.cpp headers stay out of embedded_server.hpp. Single-process mode only;
// the handle is non-owning and is not wired to real inference yet.
struct InProcessEngine {};

using libopenai::Request;
using libopenai::Response;

// ============================================================================
// EmbeddedBackend
// ============================================================================

EmbeddedBackend::EmbeddedBackend(std::string name, Config config)
    : name_(std::move(name)), config_(std::move(config)) {
    health_ = libopenai::BackendHealth::Unknown;
}

EmbeddedBackend::~EmbeddedBackend() {
    // In a real integration, llama_free(llama_ctx_) would be called here.
}

void EmbeddedBackend::record_request_start() {
    std::lock_guard<std::mutex> lock(load_mutex_);
    load_.active_requests++;
    load_.total_requests++;
}

std::future<Response> EmbeddedBackend::ExecuteAsync(const Request& req) {
    record_request_start();
    auto start = std::chrono::steady_clock::now();

    // Single-process mode: forward directly to the in-process server_context
    if (server_) {
        auto promise = std::make_shared<std::promise<Response>>();
        auto future = promise->get_future();

        std::thread([this, req, promise, start]() mutable {
            try {
                Response resp = server_->execute_internal(req);

                auto end = std::chrono::steady_clock::now();
                double latency_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();

                {
                    std::lock_guard<std::mutex> lock(load_mutex_);
                    load_.active_requests =
                        (load_.active_requests > 0) ? load_.active_requests - 1 : 0;
                    if (resp.status_code < 200 || resp.status_code >= 300)
                        load_.failed_requests++;
                    load_.avg_latency_ms =
                        (load_.avg_latency_ms * 0.9) + (latency_ms * 0.1);
                    load_.current_load =
                        static_cast<double>(load_.active_requests) / 100.0;
                    if (load_.current_load > 1.0) load_.current_load = 1.0;
                }

                promise->set_value(std::move(resp));
            } catch (const std::exception& e) {
                Response resp;
                resp.status_code = -1;
                resp.body = std::string("EmbeddedBackend error: ") + e.what();
                promise->set_value(std::move(resp));
            } catch (...) {
                Response resp;
                resp.status_code = -1;
                resp.body = "EmbeddedBackend: unknown exception";
                promise->set_value(std::move(resp));
            }
        }).detach();

        return future;
    }

    auto promise = std::make_shared<std::promise<Response>>();
    auto future = promise->get_future();

    std::thread([this, req, promise, start]() mutable {
        try {
            Response resp;
            bool ok = false;

            std::string ep = req.endpoint;
            // Accept both bare and /v1-prefixed paths
            if (ep == "/v1/models" || ep == "/models") {
                ok = process_models(resp);
            } else if (ep == "/v1/chat/completions" || ep == "/chat/completions") {
                ok = process_chat_completion(req, resp);
            } else if (ep == "/v1/completions" || ep == "/completions" ||
                       ep == "/completion") {
                ok = process_completion(req, resp);
            } else if (ep == "/v1/embeddings" || ep == "/embeddings" ||
                       ep == "/embedding") {
                ok = process_embeddings(req, resp);
            } else {
                resp.status_code = 404;
                resp.body = "{\"error\":{\"message\":\"Endpoint not found: " +
                            ep +
                            "\",\"type\":\"not_found_error\",\"code\":404}}";
                ok = true; // not an internal failure
            }

            if (!ok) {
                resp.status_code = 500;
                resp.body = "{\"error\":{\"message\":\"Backend processing failed\",\"type\":\"backend_error\",\"code\":500}}";
            }

            auto end = std::chrono::steady_clock::now();
            double latency_ms =
                std::chrono::duration<double, std::milli>(end - start).count();

            {
                std::lock_guard<std::mutex> lock(load_mutex_);
                load_.active_requests =
                    (load_.active_requests > 0) ? load_.active_requests - 1 : 0;
                if (!ok) load_.failed_requests++;
                load_.avg_latency_ms =
                    (load_.avg_latency_ms * 0.9) + (latency_ms * 0.1);
                load_.current_load =
                    static_cast<double>(load_.active_requests) / 100.0;
                if (load_.current_load > 1.0) load_.current_load = 1.0;
            }

            promise->set_value(std::move(resp));
        } catch (const std::exception& e) {
            Response resp;
            resp.status_code = -1;
            resp.body = std::string("EmbeddedBackend error: ") + e.what();
            promise->set_value(std::move(resp));
        } catch (...) {
            Response resp;
            resp.status_code = -1;
            resp.body = "EmbeddedBackend: unknown exception";
            promise->set_value(std::move(resp));
        }
    }).detach();

    return future;
}

Response EmbeddedBackend::Execute(const Request& req) {
    return ExecuteAsync(req).get();
}

libopenai::BackendHealth EmbeddedBackend::Health() const {
    if (!model_loaded_) {
        health_ = libopenai::BackendHealth::Unhealthy;
    } else if (load_.current_load > 0.9) {
        health_ = libopenai::BackendHealth::Degraded;
    } else {
        health_ = libopenai::BackendHealth::Healthy;
    }
    return health_;
}

libopenai::BackendLoad EmbeddedBackend::GetLoad() const {
    std::lock_guard<std::mutex> lock(load_mutex_);
    return load_;
}

bool EmbeddedBackend::SupportsModel(const std::string& model) const {
    (void)model;
    return true; // Embedded backend accepts any model name, routes to loaded GGUF
}

bool EmbeddedBackend::process_models(Response& resp) {
    if (!model_loaded_) {
        resp.status_code = 503;
        resp.body = "{\"error\":{\"message\":\"Model not loaded\",\"type\":\"unavailable_error\",\"code\":503}}";
        return false;
    }

    json data = json::array();
    json model_entry;
    model_entry["id"] = config_.model_path;
    model_entry["object"] = "model";
    model_entry["created"] = 0;
    model_entry["owned_by"] = "embedded";
    model_entry["root"] = config_.model_path;
    model_entry["parent"] = nullptr;
    data.push_back(model_entry);

    json models_list;
    models_list["object"] = "list";
    models_list["data"] = data;

    resp.status_code = 200;
    resp.body = models_list.dump();
    resp.json_data = models_list;
    return true;
}

bool EmbeddedBackend::process_completion(const Request& req, Response& resp) {
    // Delegate to the real server_context integration via EmbeddedServer
    if (server_) {
        resp = server_->execute_internal(req);
        return resp.status_code >= 200 && resp.status_code < 300;
    }
    if (!model_loaded_) {
        resp.status_code = 503;
        resp.body = "{\"error\":{\"message\":\"Model not loaded\",\"type\":\"unavailable_error\",\"code\":503}}";
        return false;
    }
    resp.status_code = 503;
    resp.body = "{\"error\":{\"message\":\"server not bound\",\"type\":\"unavailable_error\",\"code\":503}}";
    return false;
}

bool EmbeddedBackend::process_chat_completion(const Request& req, Response& resp) {
    // Delegate to the real server_context integration via EmbeddedServer
    if (server_) {
        resp = server_->execute_internal(req);
        return resp.status_code >= 200 && resp.status_code < 300;
    }
    if (!model_loaded_) {
        resp.status_code = 503;
        resp.body = "{\"error\":{\"message\":\"Model not loaded\",\"type\":\"unavailable_error\",\"code\":503}}";
        return false;
    }
    resp.status_code = 503;
    resp.body = "{\"error\":{\"message\":\"server not bound\",\"type\":\"unavailable_error\",\"code\":503}}";
    return false;
}

bool EmbeddedBackend::process_embeddings(const Request& req, Response& resp) {
    // Delegate to the real server_context integration via EmbeddedServer
    if (server_) {
        resp = server_->execute_internal(req);
        return resp.status_code >= 200 && resp.status_code < 300;
    }
    if (!model_loaded_) {
        resp.status_code = 503;
        resp.body = "{\"error\":{\"message\":\"Model not loaded\",\"type\":\"unavailable_error\",\"code\":503}}";
        return false;
    }
    resp.status_code = 503;
    resp.body = "{\"error\":{\"message\":\"server not bound\",\"type\":\"unavailable_error\",\"code\":503}}";
    return false;
}

// ============================================================================
// EmbeddedServer
// ============================================================================

EmbeddedServer::EmbeddedServer(Config config) : config_(std::move(config)) {}

EmbeddedServer::~EmbeddedServer() {
    stop();
}

bool EmbeddedServer::start() {
    if (running_.load()) return true;

    if (!init_server_context()) {
        std::cerr << "[EmbeddedServer] Failed to init server_context" << std::endl;
        return false;
    }

    // Build the high-level route table (server_routes) over server_context.
    // This mirrors tools/server/server.cpp::server_register_routes and gives
    // every llama-server endpoint a real server_context-backed implementation.
    routes_ = std::make_unique<server_routes>(params_, server_ctx_);
    routes_->update_meta(server_ctx_);

    server_loop_thread_ = std::thread([this]() {
        server_ctx_.start_loop();
    });

    http_server_ = std::make_unique<httplib::Server>();

    http_server_->set_default_headers({{"Access-Control-Allow-Origin", "*"},
                                       {"Access-Control-Allow-Methods",
                                        "GET, POST, DELETE, OPTIONS"},
                                       {"Access-Control-Allow-Headers", "*"}});

    register_routes();

    int bound_port;
    if (config_.port == 0) {
        bound_port = http_server_->bind_to_any_port(config_.host);
    } else {
        if (!http_server_->bind_to_port(config_.host, config_.port)) {
            std::cerr << "[EmbeddedServer] Failed to bind HTTP server to "
                      << config_.host << ":" << config_.port << std::endl;
            return false;
        }
        bound_port = config_.port;
    }

    if (bound_port < 0) {
        std::cerr << "[EmbeddedServer] Failed to bind HTTP server" << std::endl;
        return false;
    }
    config_.port = bound_port;

    http_server_thread_ = std::thread([this]() {
        http_server_->listen_after_bind();
    });

    running_.store(true);

    if (backend_) {
        backend_->set_server(this);
    }

    std::cout << "[EmbeddedServer] Started (single-process, in-memory) on "
              << config_.host << ":" << config_.port << std::endl;
    return true;
}

void EmbeddedServer::stop() {
    if (!running_.load()) return;

    // Terminate the server_context loop
    server_ctx_.terminate();

    // Join the background thread
    if (server_loop_thread_.joinable()) {
        server_loop_thread_.join();
    }

    // Stop the HTTP server
    if (http_server_) {
        http_server_->stop();
    }
    if (http_server_thread_.joinable()) {
        http_server_thread_.join();
    }

    // Clean up child process if multi-process mode
    stop_llama_server_fallback();

    running_.store(false);
    model_ready_ = false;
    std::cout << "[EmbeddedServer] Stopped" << std::endl;
}

std::shared_ptr<libopenai::InferenceClient> EmbeddedServer::create_client() {
    if (!running_.load()) {
        std::cerr << "[EmbeddedServer] Server not running, cannot create client" << std::endl;
        return nullptr;
    }

    // Create the embedded backend
    EmbeddedBackend::Config backend_config;
    backend_config.model_path = config_.model_path;
    backend_config.n_ctx = config_.n_ctx;
    backend_config.n_batch = config_.n_batch;
    backend_config.n_ubatch = config_.n_ubatch;
    backend_config.n_gpu_layers = config_.n_gpu_layers;
    backend_config.n_threads = config_.n_threads;
    backend_config.embedding = config_.embedding;
    backend_config.verbose = false;

    backend_ = std::make_shared<EmbeddedBackend>("embedded-llama", backend_config);
    backend_->set_server(this);

    // Wrap the backend as a provider and register it
    libopenai::ProviderInfo info;
    info.name = "embedded-llama";
    info.tier = libopenai::ProviderTier::Device;
    info.capabilities.supported_modalities = {libopenai::Modality::Text, libopenai::Modality::Chat, libopenai::Modality::Embedding};
    info.capabilities.supports_streaming = false;
    info.capabilities.max_context_length = config_.n_ctx > 0 ? config_.n_ctx : 4096;

    auto provider = std::make_shared<libopenai::BackendProvider>(info, backend_);

    registry_ = std::make_shared<libopenai::ProviderRegistry>();
    registry_->RegisterProvider(provider);

    // Create the client with the registry
    auto client = std::make_shared<libopenai::InferenceClient>();
    client->SetRegistry(registry_);

    return client;
}


void EmbeddedServer::register_routes() {
    auto& svr = *http_server_;

    // Stable should_stop for server_http_req (reference member, must outlive the call)
    static const std::function<bool()> should_stop_false = []() { return false; };

    auto to_server_http_req = [&](const httplib::Request& req) {
        server_http_req sreq{
            {},                                    // params
            {},                                    // headers
            req.path,                              // path
            "",                                    // query_string (httplib::Request has no query member)
            req.body,                              // body
            {},                                    // files
            should_stop_false                      // should_stop (const reference)
        };
        for (const auto& kv : req.params) {
            sreq.params[kv.first] = kv.second;
        }
        for (const auto& kv : req.headers) {
            sreq.headers[kv.first] = kv.second;
        }
        return sreq;
    };

    // Write a server_http_res_ptr into an httplib::Response (supports streaming)
    auto write_res = [&](const server_http_res_ptr& hres, httplib::Response& res) {
        res.status = static_cast<unsigned int>(hres->status);
        for (const auto& kv : hres->headers) {
            res.set_header(kv.first, kv.second);
        }
        if (hres->next) {
            // streaming: chunked provider
            res.set_chunked_content_provider(
                hres->content_type,
                [&hres](size_t offset, httplib::DataSink& sink) -> bool {
                    (void)offset;
                    std::string chunk;
                    bool more = hres->next(chunk);
                    if (!more && chunk.empty()) {
                        return false;
                    }
                    return sink.write(chunk.data(), chunk.size());
                },
                nullptr);
        } else {
            res.set_content(hres->data, hres->content_type);
        }
    };

    auto dispatch = [&](const server_http_context::handler_t& handler,
                        const httplib::Request& req, httplib::Response& res) {
        if (!handler) {
            res.status = 501;
            res.set_content("{\"error\":{\"message\":\"not implemented\",\"type\":\"not_implemented_error\",\"code\":501}}", "application/json");
            return;
        }
        try {
            auto hres = handler(to_server_http_req(req));
            write_res(hres, res);
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":{\"message\":\"" + std::string(e.what()) + "\",\"type\":\"server_error\",\"code\":500}}", "application/json");
        } catch (...) {
            res.status = 500;
            res.set_content("{\"error\":{\"message\":\"unknown error\",\"type\":\"server_error\",\"code\":500}}", "application/json");
        }
    };

    auto route = [&](const std::string& path,
                     const server_http_context::handler_t& handler) {
        auto fn = [this, &dispatch, handler](const httplib::Request& req, httplib::Response& res) {
            dispatch(handler, req, res);
        };
        return fn;
    };

    // ---- GET ----
    svr.Get("/health",              route("/health",              routes_->get_health));
    svr.Get("/v1/health",           route("/v1/health",           routes_->get_health));
    svr.Get("/metrics",             route("/metrics",             routes_->get_metrics));
    svr.Get("/props",               route("/props",               routes_->get_props));
    svr.Get("/models",              route("/models",              routes_->get_models));
    svr.Get("/v1/models",           route("/v1/models",           routes_->get_models));
    svr.Get("/lora-adapters",       route("/lora-adapters",       routes_->get_lora_adapters));
    svr.Get("/v1/lora-adapters",    route("/v1/lora-adapters",    routes_->get_lora_adapters));
    svr.Get("/slots",               route("/slots",               routes_->get_slots));
    svr.Get("/v1/slots",            route("/v1/slots",            routes_->get_slots));

    // ---- POST ----
    svr.Post("/props",              route("/props",              routes_->post_props));
    svr.Post("/completion",         route("/completion",         routes_->post_completions));
    svr.Post("/completions",        route("/completions",        routes_->post_completions));
    svr.Post("/v1/completions",     route("/v1/completions",     routes_->post_completions_oai));
    svr.Post("/chat/completions",   route("/chat/completions",   routes_->post_chat_completions));
    svr.Post("/v1/chat/completions",route("/v1/chat/completions",routes_->post_chat_completions));
    svr.Post("/v1/chat/completions/control", route("/v1/chat/completions/control", routes_->post_control));
    svr.Post("/v1/responses",       route("/v1/responses",       routes_->post_responses_oai));
    svr.Post("/responses",          route("/responses",          routes_->post_responses_oai));
    svr.Post("/v1/audio/transcriptions", route("/v1/audio/transcriptions", routes_->post_transcriptions_oai));
    svr.Post("/audio/transcriptions",    route("/audio/transcriptions",    routes_->post_transcriptions_oai));
    svr.Post("/tts",                route("/tts",                routes_->post_tts));
    svr.Post("/v1/messages",        route("/v1/messages",        routes_->post_anthropic_messages));
    svr.Post("/infill",             route("/infill",             routes_->post_infill));
    svr.Post("/embedding",          route("/embedding",          routes_->post_embeddings));
    svr.Post("/embeddings",         route("/embeddings",         routes_->post_embeddings));
    svr.Post("/v1/embeddings",      route("/v1/embeddings",      routes_->post_embeddings_oai));
    svr.Post("/rerank",             route("/rerank",             routes_->post_rerank));
    svr.Post("/reranking",          route("/reranking",          routes_->post_rerank));
    svr.Post("/v1/rerank",          route("/v1/rerank",          routes_->post_rerank));
    svr.Post("/v1/reranking",       route("/v1/reranking",       routes_->post_rerank));
    svr.Post("/v1/systemone",       route("/v1/systemone",       routes_->post_systemone));
    svr.Post("/tokenize",           route("/tokenize",           routes_->post_tokenize));
    svr.Post("/detokenize",         route("/detokenize",         routes_->post_detokenize));
    svr.Post("/apply-template",     route("/apply-template",     routes_->post_apply_template));
    svr.Post("/chat/completions/input_tokens",        route("/chat/completions/input_tokens",        routes_->post_chat_completions_tok));
    svr.Post("/v1/chat/completions/input_tokens",     route("/v1/chat/completions/input_tokens",     routes_->post_chat_completions_tok));
    svr.Post("/responses/input_tokens",               route("/responses/input_tokens",               routes_->post_responses_tok_oai));
    svr.Post("/v1/responses/input_tokens",            route("/v1/responses/input_tokens",            routes_->post_responses_tok_oai));
    svr.Post("/v1/messages/count_tokens",             route("/v1/messages/count_tokens",             routes_->post_anthropic_count_tokens));
    svr.Post("/lora-adapters",      route("/lora-adapters",      routes_->post_lora_adapters));
    svr.Post("/slots",              route("/slots",              routes_->post_slots));

    svr.Options(R"(.*/)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    svr.Options(R"(.*/)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });
}

void EmbeddedServer::handle_route(const std::string& handler_name,
                                  const httplib::Request& req,
                                  httplib::Response& res) {
    // Dispatcher kept for API compatibility. Routes are registered directly
    // against routes_ handlers in register_routes(); this path is used when
    // a handler is invoked by name.
    static const std::function<bool()> should_stop_false = []() { return false; };
    server_http_req sreq{
        {},                                    // params
        {},                                    // headers
        req.path,                              // path
        "",                                    // query_string
        req.body,                              // body
        {},                                    // files
        should_stop_false                      // should_stop
    };
    for (const auto& kv : req.params) {
        sreq.params[kv.first] = kv.second;
    }
    for (const auto& kv : req.headers) {
        sreq.headers[kv.first] = kv.second;
    }

    server_http_context::handler_t handler = nullptr;
    if (handler_name == "get_health") handler = routes_->get_health;
    else if (handler_name == "get_metrics") handler = routes_->get_metrics;
    else if (handler_name == "get_props") handler = routes_->get_props;
    else if (handler_name == "post_props") handler = routes_->post_props;
    else if (handler_name == "get_models") handler = routes_->get_models;
    else if (handler_name == "post_completions") handler = routes_->post_completions;
    else if (handler_name == "post_completions_oai") handler = routes_->post_completions_oai;
    else if (handler_name == "post_chat_completions") handler = routes_->post_chat_completions;
    else if (handler_name == "post_control") handler = routes_->post_control;
    else if (handler_name == "post_responses_oai") handler = routes_->post_responses_oai;
    else if (handler_name == "post_transcriptions_oai") handler = routes_->post_transcriptions_oai;
    else if (handler_name == "post_tts") handler = routes_->post_tts;
    else if (handler_name == "post_anthropic_messages") handler = routes_->post_anthropic_messages;
    else if (handler_name == "post_infill") handler = routes_->post_infill;
    else if (handler_name == "post_embeddings") handler = routes_->post_embeddings;
    else if (handler_name == "post_embeddings_oai") handler = routes_->post_embeddings_oai;
    else if (handler_name == "post_rerank") handler = routes_->post_rerank;
    else if (handler_name == "post_systemone") handler = routes_->post_systemone;
    else if (handler_name == "post_tokenize") handler = routes_->post_tokenize;
    else if (handler_name == "post_detokenize") handler = routes_->post_detokenize;
    else if (handler_name == "post_apply_template") handler = routes_->post_apply_template;
    else if (handler_name == "post_chat_completions_tok") handler = routes_->post_chat_completions_tok;
    else if (handler_name == "post_responses_tok_oai") handler = routes_->post_responses_tok_oai;
    else if (handler_name == "post_anthropic_count_tokens") handler = routes_->post_anthropic_count_tokens;
    else if (handler_name == "get_lora_adapters") handler = routes_->get_lora_adapters;
    else if (handler_name == "post_lora_adapters") handler = routes_->post_lora_adapters;
    else if (handler_name == "get_slots") handler = routes_->get_slots;
    else if (handler_name == "post_slots") handler = routes_->post_slots;

    if (!handler) {
        res.status = 501;
        res.set_content("{\"error\":{\"message\":\"handler not implemented\",\"type\":\"not_implemented_error\",\"code\":501}}", "application/json");
        return;
    }
    try {
        auto hres = handler(sreq);
        res.status = static_cast<unsigned int>(hres->status);
        res.set_content(hres->data, hres->content_type);
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content("{\"error\":{\"message\":\"" + std::string(e.what()) + "\",\"type\":\"server_error\",\"code\":500}}", "application/json");
    } catch (...) {
        res.status = 500;
        res.set_content("{\"error\":{\"message\":\"unknown error\",\"type\":\"server_error\",\"code\":500}}", "application/json");
    }
    }


void EmbeddedServer::handle_health(const httplib::Request& req, httplib::Response& res) {
    handle_route("get_health", req, res);
}
void EmbeddedServer::handle_metrics(const httplib::Request& req, httplib::Response& res) {
    handle_route("get_metrics", req, res);
}
void EmbeddedServer::handle_props(const httplib::Request& req, httplib::Response& res) {
    handle_route("get_props", req, res);
}
void EmbeddedServer::handle_models(const httplib::Request& req, httplib::Response& res) {
    handle_route("get_models", req, res);
}
void EmbeddedServer::handle_completions(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_completions", req, res);
}
void EmbeddedServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_chat_completions", req, res);
}
void EmbeddedServer::handle_embeddings(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_embeddings", req, res);
}
void EmbeddedServer::handle_transcriptions(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_transcriptions_oai", req, res);
}
void EmbeddedServer::handle_tts(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_tts", req, res);
}
void EmbeddedServer::handle_realtime(const httplib::Request& req, httplib::Response& res) {
    // WebSocket upgrade is handled directly in register_routes(); this HTTP
    // fallback exists only for API compatibility.
    res.status = 501;
    res.set_content("{\"error\":{\"message\":\"use WebSocket /v1/realtime\",\"type\":\"not_implemented_error\",\"code\":501}}", "application/json");
}
void EmbeddedServer::handle_anthropic_messages(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_anthropic_messages", req, res);
}
void EmbeddedServer::handle_infill(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_infill", req, res);
}
void EmbeddedServer::handle_control(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_control", req, res);
}
void EmbeddedServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_responses_oai", req, res);
}
void EmbeddedServer::handle_tokenize(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_tokenize", req, res);
}
void EmbeddedServer::handle_detokenize(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_detokenize", req, res);
}
void EmbeddedServer::handle_apply_template(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_apply_template", req, res);
}
void EmbeddedServer::handle_rerank(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_rerank", req, res);
}
void EmbeddedServer::handle_systemone(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_systemone", req, res);
}
void EmbeddedServer::handle_token_counting(const httplib::Request& req, httplib::Response& res) {
    handle_route("post_anthropic_count_tokens", req, res);
}
void EmbeddedServer::handle_lora_adapters(const httplib::Request& req, httplib::Response& res) {
    handle_route("get_lora_adapters", req, res);
}
void EmbeddedServer::handle_slots(const httplib::Request& req, httplib::Response& res) {
    handle_route("get_slots", req, res);
}
void EmbeddedServer::handle_images(const httplib::Request& req, httplib::Response& res) {
    // Multimodal images are served through server_context slots (mtmd) inside
    // post_chat_completions / post_transcriptions_oai. No separate endpoint.
    res.status = 501;
    res.set_content("{\"error\":{\"message\":\"images served via chat completions\",\"type\":\"not_implemented_error\",\"code\":501}}", "application/json");
}

bool EmbeddedServer::init_server_context() {
    if (model_ready_) return true;

    // Initialize the llama backend once
    static std::atomic<bool> backend_initialized{false};
    if (!backend_initialized.exchange(true)) {
        common_init();
        llama_backend_init();
    }

    // Build common_params from config
    params_ = common_params{};
    params_.model.path = config_.model_path;
    params_.n_ctx = config_.n_ctx;
    params_.n_batch = config_.n_batch;
    params_.n_ubatch = config_.n_ubatch;
    params_.n_gpu_layers = config_.n_gpu_layers;
    params_.cpuparams.n_threads = config_.n_threads;
    params_.embedding = config_.embedding;
    params_.n_parallel = config_.n_parallel;
    params_.kv_unified = config_.kv_unified;
    params_.cont_batching = config_.cont_batching;
    params_.use_jinja = config_.jinja;
    params_.chat_template = config_.chat_template;
    if (!config_.model_alias.empty()) {
        params_.model_alias = std::set<std::string>({config_.model_alias});
    }
    params_.endpoint_slots = config_.endpoint_slots;
    params_.endpoint_props = config_.endpoint_props;
    params_.endpoint_metrics = config_.endpoint_metrics;
    params_.sleep_idle_seconds = static_cast<float>(config_.timeout_seconds);

    // Multimodal projector (mtmd) - passed straight to server_context which
    // loads the projector through mtmd on load_model(). A valid mmproj enables
    // image/audio/video chat parts, /v1/audio/transcriptions, /tts and /v1/realtime.
    if (!config_.no_mmproj && !config_.mmproj_path.empty()) {
        params_.mmproj.path = config_.mmproj_path;
        params_.mmproj_use_gpu = config_.mmproj_use_gpu;
        // mmproj_device defaults to nullptr (CPU). GPU offload is controlled
        // by mmproj_use_gpu; device selection can be added later via
        // ggml_backend APIs when a specific device name is required.
    }

    // Configure speculative decoding via common_params.speculative
    // server_context handles draft model loading and speculative decoding internally
    if (!config_.draft_model_path.empty()) {
        params_.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
        params_.speculative.draft.n_max = config_.n_draft;
        params_.speculative.draft.n_min = 0;
        params_.speculative.draft.p_split = 0.1f;
        params_.speculative.draft.p_min = 0.0f;
        params_.speculative.draft.backend_sampling = true;
        params_.speculative.draft.probabilistic = false;
        params_.speculative.draft.mparams.path = config_.draft_model_path;
        params_.speculative.draft.n_gpu_layers = config_.n_gpu_layers;
        params_.speculative.draft.n_ubatch = config_.n_ubatch;
        params_.speculative.draft.cache_type_k = GGML_TYPE_F16;
        params_.speculative.draft.cache_type_v = GGML_TYPE_F16;
        params_.speculative.draft.cpuparams.n_threads = config_.n_threads;
        params_.speculative.draft.cpuparams_batch.n_threads = config_.n_threads;
    }

    // Load the model using server_context (high-level integration)
    // This replaces raw llama.h API calls and handles speculative decoding internally
    if (!server_ctx_.load_model(params_)) {
        std::cerr << "[EmbeddedServer] Failed to load model via server_context: "
                  << config_.model_path << std::endl;
        return false;
    }

    model_ready_ = true;
    std::cout << "[EmbeddedServer] Model loaded via server_context: model="
              << config_.model_path << std::endl;

    if (!config_.draft_model_path.empty()) {
        std::cout << "[EmbeddedServer] Speculative decoding configured with draft model: "
                  << config_.draft_model_path << std::endl;
    }

    // mtmd multimodal projector is loaded by server_context inside load_model()
    // when params_.mmproj.path is set (see server-context.cpp). No direct
    // mtmd_* call is needed here; server_context owns mtmd_ctx_ and routes
    // multimodal chunks through slots. Query status via get_meta().has_mtmd.

    return true;
}

void EmbeddedServer::free_model() {
    // server_context owns the model and context; terminate will clean up
    model_ready_ = false;
}

libopenai::Response EmbeddedServer::execute_internal(const libopenai::Request& req) {
    Response resp;
    resp.status_code = 503;
    resp.body = "{\"error\":{\"message\":\"model context not ready\",\"type\":\"unavailable_error\",\"code\":503}}";

    if (!model_ready_) {
        return resp;
    }

    std::string ep = req.endpoint;
    // Normalize: strip /v1 prefix for matching
    std::string bare = ep;
    if (bare.size() > 3 && bare.substr(0, 3) == "/v1") bare = bare.substr(3);

    // Route through server_routes handlers - every endpoint served by
    // server_context, no 501 fall-through for registered routes.
    auto call_route = [&](const server_http_context::handler_t& handler) -> Response {
        if (!handler) {
            Response r;
            r.status_code = 501;
            r.body = "{\"error\":{\"message\":\"not implemented\",\"type\":\"not_implemented_error\",\"code\":501}}";
            return r;
        }
        static const std::function<bool()> should_stop_false = []() { return false; };
        server_http_req sreq{
            {},                                    // params
            {},                                    // headers
            ep,                                    // path
            "",                                    // query_string
            req.payload,                           // body
            {},                                    // files
            should_stop_false                      // should_stop
        };
        for (const auto& kv : req.headers) {
            sreq.headers[kv.first] = kv.second;
        }
        auto hres = handler(sreq);
        Response r;
        r.status_code = hres->status;
        r.body = hres->data;
        try {
            r.json_data = json::parse(hres->data);
        } catch (...) {
            // not JSON, leave json_data empty
        }
        return r;
    };

    // GET /models
    if (bare == "/models" && req.method == "GET") {
        return call_route(routes_->get_models);
    }

    // POST endpoints - route through server_routes
    if (req.method == "POST") {
        if (bare == "/completions" || bare == "/completion") {
            return call_route(routes_->post_completions);
        }
        if (bare == "/v1/completions") {
            return call_route(routes_->post_completions_oai);
        }
        if (bare == "/chat/completions") {
            return call_route(routes_->post_chat_completions);
        }
        if (bare == "/v1/chat/completions") {
            return call_route(routes_->post_chat_completions);
        }
        if (bare == "/v1/chat/completions/control") {
            return call_route(routes_->post_control);
        }
        if (bare == "/v1/responses" || bare == "/responses") {
            return call_route(routes_->post_responses_oai);
        }
        if (bare == "/v1/audio/transcriptions" || bare == "/audio/transcriptions") {
            return call_route(routes_->post_transcriptions_oai);
        }
        if (bare == "/tts") {
            return call_route(routes_->post_tts);
        }
        if (bare == "/v1/messages") {
            return call_route(routes_->post_anthropic_messages);
        }
        if (bare == "/infill") {
            return call_route(routes_->post_infill);
        }
        if (bare == "/embedding" || bare == "/embeddings" || bare == "/v1/embeddings") {
            return call_route(routes_->post_embeddings);
        }
        if (bare == "/rerank" || bare == "/reranking" || bare == "/v1/rerank" || bare == "/v1/reranking") {
            return call_route(routes_->post_rerank);
        }
        if (bare == "/v1/systemone") {
            return call_route(routes_->post_systemone);
        }
        if (bare == "/tokenize") {
            return call_route(routes_->post_tokenize);
        }
        if (bare == "/detokenize") {
            return call_route(routes_->post_detokenize);
        }
        if (bare == "/apply-template") {
            return call_route(routes_->post_apply_template);
        }
        if (bare == "/chat/completions/input_tokens" || bare == "/v1/chat/completions/input_tokens") {
            return call_route(routes_->post_chat_completions_tok);
        }
        if (bare == "/responses/input_tokens" || bare == "/v1/responses/input_tokens") {
            return call_route(routes_->post_responses_tok_oai);
        }
        if (bare == "/v1/messages/count_tokens") {
            return call_route(routes_->post_anthropic_count_tokens);
        }
        if (bare == "/lora-adapters") {
            return call_route(routes_->post_lora_adapters);
        }
        if (bare == "/slots") {
            return call_route(routes_->post_slots);
        }
    }

    // Unknown endpoint
    resp.status_code = 404;
    resp.body = "{\"error\":{\"message\":\"Endpoint not found: " + ep + "\",\"type\":\"not_found_error\",\"code\":404}}";
    return resp;
}

void EmbeddedServer::stop_llama_server_fallback() {
    if (!llama_server_fallback_) return;

#ifdef _WIN32
    if (llama_server_handle_) {
        TerminateProcess(llama_server_handle_, 0);
        CloseHandle(llama_server_handle_);
        llama_server_handle_ = nullptr;
    }
#else
    if (llama_server_pid_ > 0) {
        kill(llama_server_pid_, SIGTERM);
        waitpid(llama_server_pid_, nullptr, 0);
        llama_server_pid_ = 0;
    }
#endif
    llama_server_fallback_ = false;
}

} // namespace embedded
























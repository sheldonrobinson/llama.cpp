#include "server-common.h"
#include "server-embedded.h"
#include "server-context.h"

#include "preset.h"
#include "download.h"

#include "uv-memory-server.hpp" // TODO: remove this once we use HTTP client from download.h
#include "server-model-manager.hpp"
#include <functional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_READY "cmd_child_to_router:ready"

static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

//
// server_models
//
server_models::server_models(
        const common_params & params)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params) {
				  
	// set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = "llama.cpp.embedded";
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using default as fallback: %s\n", bin_path.c_str());
    }
    load_models();
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }
    meta.update_args(ctx_preset, bin_path); // render args
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* th      */ std::thread(),
        /* meta    */ std::move(meta)
    };
}

// TODO: allow refreshing cached model list
void server_models::load_models() {
    // loading models from 3 sources:
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_INF("Loaded %zu cached model presets\n", cached_models.size());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_INF("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_INF("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
    }
    for (const auto & [name, preset] : local_models) {
        final_presets[name] = preset;
    }

    // process custom presets from INI
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            // apply custom config if exists
            common_preset & target = final_presets[name];
            target.merge(custom);
        } else {
            // otherwise add directly
            final_presets[name] = custom;
        }
    }

    // convert presets to server_model_meta and add to mapping
    for (const auto & preset : final_presets) {
        server_model_meta meta{
            /* preset       */ preset.second,
            /* name         */ preset.first,
            /* status       */ server_model_status::SERVER_MODEL_STATUS_UNLOADED,
            /* last_used    */ 0,
            /* args         */ std::vector<std::string>(),
            /* exit_code    */ 0,
            /* stop_timeout */ DEFAULT_STOP_TIMEOUT,
        };
        add_model(std::move(meta));
    }

    // log available models
    {
        std::unordered_set<std::string> custom_names;
        for (const auto & [name, preset] : custom_presets) {
            custom_names.insert(name);
        }
        SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
        for (const auto & [name, inst] : mapping) {
            bool has_custom = custom_names.find(name) != custom_names.end();
            SRV_INF("  %c %s\n", has_custom ? '*' : ' ', name.c_str());
        }
    }

    // handle custom stop-timeout option
    for (auto & [name, inst] : mapping) {
        std::string val;
        if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
            try {
                inst.meta.stop_timeout = std::stoi(val);
            } catch (...) {
                SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                    val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
            }
        }
    }

    // load any autoload models
    std::vector<std::string> models_to_load;
    for (const auto & [name, inst] : mapping) {
        std::string val;
        if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val)) {
            models_to_load.push_back(name);
        }
    }
    if ((int)models_to_load.size() > base_params.models_max) {
        throw std::runtime_error(string_format(
            "number of models to load on startup (%zu) exceeds models_max (%d)",
            models_to_load.size(),
            base_params.models_max
        ));
    }
    for (const auto & name : models_to_load) {
        SRV_INF("(startup) loading model %s\n", name.c_str());
        load(name);
    }
}


void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify 
	wait_until_loaded(name);
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    return mapping.find(name) != mapping.end();
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    return std::nullopt;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    std::string lru_model_name = "";
    int64_t lru_last_used = ggml_time_ms();
    size_t count_active = 0;
    {
        std::unique_lock<std::mutex> lk(mutex);
        for (const auto & m : mapping) {
            if (m.second.meta.is_active()) {
                count_active++;
                if (m.second.meta.last_used < lru_last_used) {
                    lru_model_name = m.first;
                    lru_last_used = m.second.meta.last_used;
                }
            }
        }
    }
    if (!lru_model_name.empty() && count_active >= (size_t)base_params.models_max) {
        SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
        unload(lru_model_name);
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_model_name]() {
                return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}

static ModelManager g_modelManager(16384); // 16 GB limit

void server_models::load(const std::string & name) {
    if (!has_model(name)) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    unload_lru();

    std::lock_guard<std::mutex> lk(mutex);

    auto meta = mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // prepare new instance info
    instance_t inst;
    inst.meta           = meta;
    inst.meta.status    = server_model_status::SERVER_MODEL_STATUS_LOADING;
    inst.meta.last_used = ggml_time_ms();
	
	{
        SRV_INF("creating server instance with name=%s\n", inst.meta.name.c_str());

        inst.meta.update_args(ctx_preset, bin_path); // render args
		
		try {
			g_modelManager.loadModel(name, base_params);
		} catch(...){
			throw std::runtime_error("failed to load model");
		}
    }
	mapping[name] = std::move(inst);
    cv.notify_all();
}

//
// server_models_routes
//

static void res_ok(std::unique_ptr<server_core_res> & res, const json & response_data) {
    res->status = SERVER_CORE_STATUS_SUCCESS;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_core_res> & res, const json & error_data) {
    res->status = SERVER_CORE_STATUS_FAILURE;
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(const std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_core_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response("model not found", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    if (models_autoload) {
        models.ensure_model_loaded(name);
    } else {
        if (meta->status != SERVER_MODEL_STATUS_LOADED) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return false;
        }
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_core_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}


void server_models::unload(const std::string & name) {
    auto it = mapping.find(name);
    if (it != mapping.end()) {
		auto & inst = it->second;
        if (inst.meta.is_active()) {
            SRV_INF("unloading model instance name=%s\n", name.c_str());
            try{
				g_modelManager.unloadModel(name);
				update_status(name,  server_model_status::SERVER_MODEL_STATUS_UNLOADED, 0);
			}catch(...){
				update_status(name,  inst.meta.status, -1);
			}
			mapping[name] = std::move(inst);
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not loaded\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
		for (auto & [name, inst] : mapping) {
            if (inst.meta.is_active()) {
				SRV_INF("unloading model instance name=%s\n", name.c_str());
				try{
					g_modelManager.unloadModel(name);
					update_status(name,  server_model_status::SERVER_MODEL_STATUS_UNLOADED, 0);
				}catch(...){
					update_status(name,  inst.meta.status, -1);
				}
			} else {
				SRV_WRN("model instance name=%s is not loaded\n", name.c_str());
			}
		}
}

void server_models::update_status(const std::string & name, server_model_status status, int exit_code) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.status    = status;
        meta.exit_code = exit_code;
    }
    cv.notify_all();
}

void server_models::wait_until_loaded(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [this, &name]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return it->second.meta.status != SERVER_MODEL_STATUS_LOADING;
        }
        return false;
    });
}

bool server_models::ensure_model_loaded(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->status == SERVER_MODEL_STATUS_LOADED) {
        return false; // already loaded
    }
    if (meta->status == SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
        load(name);
    }

    // for loading state
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    wait_until_loaded(name);

    // check final status
    meta = get_meta(name);
    if (!meta.has_value() || meta->is_failed()) {
        throw std::runtime_error("model name=" + name + " failed to load");
    }

    return true;
}

void server_models_routes::init_routes() {
    this->get_router_props = [this](const server_core_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_core_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",          "router"},
                {"max_instances", 4}, // dummy value for testing
                // this is a dummy response to make sure webui doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_core_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_core_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_core_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_core_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, true); // update last usage for POST request only
    };

    this->post_router_models_load = [this](const server_core_req & req) {
        auto res = std::make_unique<server_core_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (model->status == SERVER_MODEL_STATUS_LOADED) {
            res_err(res, format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_core_req &) {
        auto res = std::make_unique<server_core_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }
            models_json.push_back(json {
                {"id",       meta.name},
                {"object",   "model"},    // for OAI-compat
                {"owned_by", "llamacpp"}, // for OAI-compat
                {"created",  t},          // for OAI-compat
                {"status",   status},
                // TODO: add other fields, may require reading GGUF metadata
            });
        }
        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_unload = [this](const server_core_req & req) {
        auto res = std::make_unique<server_core_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_active()) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(name);
        res_ok(res, {{"success", true}});
        return res;
    };
}

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_core_context::handler_t ex_wrapper(server_core_context::handler_t func) {
    return [func = std::move(func)](const server_core_req & req) -> server_core_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_core_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

void server_embedded_start(const common_params& args, server_status_callback* callback){
	
	common_params params = std::move(args);
	// validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    // see https://github.com/ggml-org/llama.cpp/issues/12836
    if (params.embedding && params.n_batch > params.n_ubatch) {
        LOG_WRN("%s: embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", __func__, params.n_batch, params.n_ubatch);
        LOG_WRN("%s: setting n_batch = n_ubatch = %d to avoid assertion failure\n", __func__, params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);

        params.n_parallel = 4;
        params.kv_unified = true;
    }
	
	// for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias = params.model.name;
    }

    common_init();
	
	// struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

	server_core_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return;
    }
	
	// register API routes
    server_routes routes(params, ctx_server);
	
	ctx_http.get ("/health",              ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",           ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",             ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",               ex_wrapper(routes.get_props));
    ctx_http.post("/props",               ex_wrapper(routes.post_props));
    ctx_http.post("/api/show",            ex_wrapper(routes.get_api_show));
    ctx_http.get ("/models",              ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",           ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/api/tags",            ex_wrapper(routes.get_models)); // ollama specific endpoint. public endpoint (no API key check)
    ctx_http.post("/completion",          ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",         ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",    ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat",            ex_wrapper(routes.post_chat_completions)); // ollama specific endpoint
    ctx_http.post("/v1/messages",         ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    ctx_http.post("/infill",              ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",           ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",          ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",       ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",              ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",        ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",            ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",          ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",      ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",       ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",       ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",               ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",      ex_wrapper(routes.post_slots));
	
	
	// setup clean up function, to be called before exit
    std::function<void()> clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
    };
	
	
	// start the HTTP server before loading the model to be able to serve /health requests
	if (!ctx_http.start()) {
		clean_up();
		LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
		return;
	}

	// load the model
	LOG_INF("%s: loading model\n", __func__);

	if (!ctx_server.load_model(params)) {
		clean_up();
		if (ctx_http.thread.joinable()) {
			ctx_http.thread.join();
		}
		LOG_ERR("%s: exiting due to model loading error\n", __func__);
		return;
	}

	routes.update_meta(ctx_server);
	ctx_http.is_ready.store(true);

	LOG_INF("%s: model loaded\n", __func__);
	
	shutdown_handler = [&](int) {
		// this will unblock start_loop()
		ctx_server.terminate();
	};
	
	// starting
	is_terminating.clear();
	
	// this call blocks the main thread until queue_tasks.terminate() is called
	ctx_server.start_loop();

	clean_up();
	if (ctx_http.thread.joinable()) {
		ctx_http.thread.join();
	}
}


void server_embedded_stop(){
	if (is_terminating.test_and_set()) {
        return;
    }
	if(shutdown_handler != nullptr){
		shutdown_handler(0);
	}
}
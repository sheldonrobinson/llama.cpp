#include "server-common.h"
#include "server-embedded.h"
#include "server-context.h"

#include "preset.h"
#include "download.h"

#include "uv-memory-server.hpp"
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

#if defined(__linux__) && !defined(__ANDROID__)  // Linux
#include <sys/sysinfo.h>
#endif

#if defined(__ANDROID__)  // Android
#include <fstream>
#include <sstream>
#include <string>
#endif


#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_READY "cmd_child_to_router:ready"

struct MemoryInfo {
    unsigned long long total_physical;
    unsigned long long available_physical;
};

MemoryInfo get_memory_info() {
    MemoryInfo info{};

#if defined(_WIN32)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (!GlobalMemoryStatusEx(&statex)) {
        throw std::runtime_error("Failed to get memory info on Windows");
    }
    info.total_physical     = statex.ullTotalPhys;
    info.available_physical = statex.ullAvailPhys;

#elif defined(__linux__) && !defined(__ANDROID__)
    struct sysinfo memInfo;
    if (sysinfo(&memInfo) != 0) {
        throw std::runtime_error("Failed to get memory info on Linux");
    }
    info.total_physical     = static_cast<unsigned long long>(memInfo.totalram) * memInfo.mem_unit;
    info.available_physical = static_cast<unsigned long long>(memInfo.freeram) * memInfo.mem_unit;

#elif defined(__APPLE__)
    // Get total RAM
    int      mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t total;
    size_t   len = sizeof(total);
    if (sysctl(mib, 2, &total, &len, nullptr, 0) != 0) {
        throw std::runtime_error("Failed to get total memory on macOS");
    }
    info.total_physical = total;

    // Get available RAM using Mach API
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    vm_statistics64_data_t vmstat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO, reinterpret_cast<host_info64_t>(&vmstat), &count) !=
        KERN_SUCCESS) {
        throw std::runtime_error("Failed to get available memory on macOS");
    }
    uint64_t free_mem       = static_cast<uint64_t>(vmstat.free_count) * sysconf(_SC_PAGESIZE);
    uint64_t inactive_mem   = static_cast<uint64_t>(vmstat.inactive_count) * sysconf(_SC_PAGESIZE);
    info.available_physical = free_mem + inactive_mem;

#elif defined(__ANDROID__)
    // Read from /proc/meminfo
    std::ifstream meminfo_file("/proc/meminfo");
    if (!meminfo_file.is_open()) {
        throw std::runtime_error("Failed to open /proc/meminfo on Android");
    }

    std::string        line;
    unsigned long long mem_total_kb = 0, mem_available_kb = 0;
    while (std::getline(meminfo_file, line)) {
        std::istringstream iss(line);
        std::string        key;
        unsigned long long value;
        std::string        unit;
        if (iss >> key >> value >> unit) {
            if (key == "MemTotal:") {
                mem_total_kb = value;
            } else if (key == "MemAvailable:") {
                mem_available_kb = value;
            }
        }
    }
    meminfo_file.close();

    if (mem_total_kb == 0 || mem_available_kb == 0) {
        throw std::runtime_error("Failed to parse memory info on Android");
    }

    info.total_physical     = mem_total_kb * 1024ULL;
    info.available_physical = mem_available_kb * 1024ULL;

#endif

    return info;
}

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

void server_model_meta::update_args(common_preset_context & ctx_presets, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_presets, "LLAMA_ARG_ALIAS", name);
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);
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
	
	// check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
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
			/* aliases      */ {},
            /* tags         */ {},
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
		auto join_set = [](const std::set<std::string> & s) {
            std::string result;
            for (const auto & v : s) {
                if (!result.empty()) {
                    result += ", ";
                }
                result += v;
            }
            return result;
        };
        SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
        for (const auto & [name, inst] : mapping) {
            bool has_custom = custom_names.find(name) != custom_names.end();
			std::string info;
            if (!inst.meta.aliases.empty()) {
                info += " (aliases: " + join_set(inst.meta.aliases) + ")";
            }
            if (!inst.meta.tags.empty()) {
                info += " [tags: " + join_set(inst.meta.tags) + "]";
            }
            SRV_INF("  %c %s%s\n", has_custom ? '*' : ' ', name.c_str(), info.c_str());
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
            if (common_arg_utils::is_truthy(val)) {
                models_to_load.push_back(name);
            }
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
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
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

static ModelManager g_modelManager(4096); // 4GB limit

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
			update_status(name,  server_model_status::SERVER_MODEL_STATUS_UNLOADED, -1);
			SRV_ERR("failed to load model %s\n", name.c_str());
			return;
		}

        inst.th = std::thread([this, name]() {
            ModelContext model_ctx  = g_modelManager.getModelContext(name);
            std::shared_ptr<server_context> server_ctx = model_ctx.server_ctx;
            server_ctx->start_loop();
        });
        inst.th.detach();
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

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_core_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
	// resolve alias to canonical model name
    name = meta->name;
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
                SRV_ERR("failed to unload model instance name=%s\n", name.c_str());
				update_status(name, server_model_status::SERVER_MODEL_STATUS_INVALID, -1);
			}
            if (inst.th.joinable()) {
                inst.th.join();
            }
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not loaded\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::vector<std::thread> to_join;
	for (auto & [name, inst] : mapping) {
        if (inst.meta.is_active()) {
			SRV_INF("unloading model instance name=%s\n", name.c_str());
			try{
				g_modelManager.unloadModel(name);
				update_status(name,  server_model_status::SERVER_MODEL_STATUS_UNLOADED, 0);
			}catch(...){
                SRV_ERR("failed to unload model instance name=%s\n", name.c_str());
                update_status(name, server_model_status::SERVER_MODEL_STATUS_INVALID, -1);
			}
            
            // moving the thread to join list to avoid deadlock
            to_join.push_back(std::move(inst.th));
		} else {
			SRV_WRN("model instance name=%s is not loaded\n", name.c_str());
		}
	}
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
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
        return true; // already loaded
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

server_core_res_ptr server_models::proxy_request(const server_core_req & req,
                                                 const std::string &     method,
                                                 const std::string &     name,
                                                 bool                    update_last_used) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->status != SERVER_MODEL_STATUS_LOADED) {
        throw std::invalid_argument("model name=" + name + " is not loaded");
    }
    if (update_last_used) {
        std::unique_lock<std::mutex> lk(mutex);
        mapping[name].meta.last_used = ggml_time_ms();
    }
	
	std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy =
        std::make_unique<server_core_proxy>(
			method,
			req.path,
			req.query_string,
			req.metadata,
			req.body,
			req.should_stop,
			base_params.timeout_read,
			base_params.timeout_write);
    return proxy;
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
         auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->status == SERVER_MODEL_STATUS_LOADED) {
            res_err(res, format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
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
				{"aliases",  meta.aliases},
                {"tags",     meta.tags},
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
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };
}

static std::function<void(int)> shutdown_handler;
static std::atomic_flag g_is_terminating = ATOMIC_FLAG_INIT;
static std::atomic<bool> g_is_interrupted = false;

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

static std::unordered_map<std::string, server_core_context*> g_servers;

std::string server_embedded_model_list() {
	return g_modelManager.listModelsJson().dump();
	// if(models){
		// auto list_of_models = g_modelManager.listModelsJson().dump().c_str();
		// int len = strlen(list_of_models);
		// models = (char*) calloc(len+1, sizeof(char));
		// std::memcpy(models, list_of_models, sizeof(char)*len);
		// models[len] = '\0';
	// }
}

void server_embedded_inference_svc(common_params params) {
    // common_params params = args;
    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    // see https://github.com/ggml-org/llama.cpp/issues/12836
    if (params.embedding && params.n_batch > params.n_ubatch) {
        LOG_WRN("%s: embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", __func__, params.n_batch,
                params.n_ubatch);
        LOG_WRN("%s: setting n_batch = n_ubatch = %d to avoid assertion failure\n", __func__, params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);
        params.n_parallel = 4;
        params.kv_unified = true;
    }
	std::filesystem::path modelPath(params.model.path);
	std::string modelfilename = modelPath.filename().stem().generic_string();
    // for consistency between server router mode and single-model mode, we set the same model name as alias
    // if (params.model_alias.empty() && !params.model.name.empty()) {
        // params.model_alias = params.model.name;
    // }
	params.model.name = modelfilename;
	params.model_alias.insert(modelfilename);
	

    if (g_modelManager.getModelState(modelfilename) == server_model_status::SERVER_MODEL_STATUS_UNLOADED)
	{
		g_modelManager.loadModel(modelfilename, params);
	}

    // struct that contains llama context and inference
    ModelContext model_ctx = g_modelManager.getModelContext(modelfilename);
	if(model_ctx.state == server_model_status::SERVER_MODEL_STATUS_UNLOADED) {
		return;
	}

    server_core_context ctx_http; 

    auto & result = g_servers.emplace(std::make_pair(modelfilename, &ctx_http));

    if (!result.second) {
        return;
    }
    
    if (!ctx_http.init(params)) {
        g_servers.erase(modelfilename);
        return;
    }

    // register API routes
    server_routes routes(params, *model_ctx.server_ctx);

    ctx_http.get("/health", ex_wrapper(routes.get_health));     // public endpoint (no API key check)
    ctx_http.get("/v1/health", ex_wrapper(routes.get_health));  // public endpoint (no API key check)
    ctx_http.get("/metrics", ex_wrapper(routes.get_metrics));
    ctx_http.get("/props", ex_wrapper(routes.get_props));
    ctx_http.post("/props", ex_wrapper(routes.post_props));
    ctx_http.post("/api/show", ex_wrapper(routes.get_api_show));
    ctx_http.get("/models", ex_wrapper(routes.get_models));     // public endpoint (no API key check)
    ctx_http.get("/v1/models", ex_wrapper(routes.get_models));  // public endpoint (no API key check)
    ctx_http.get("/api/tags",
                 ex_wrapper(routes.get_models));  // ollama specific endpoint. public endpoint (no API key check)
    ctx_http.post("/completion", ex_wrapper(routes.post_completions));  // legacy
    ctx_http.post("/completions", ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions", ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat", ex_wrapper(routes.post_chat_completions));       // ollama specific endpoint
	ctx_http.post("/v1/responses",        ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",           ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/messages", ex_wrapper(routes.post_anthropic_messages));  // anthropic messages API
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens));              // anthropic token counting
    ctx_http.post("/infill", ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding", ex_wrapper(routes.post_embeddings));            // legacy
    ctx_http.post("/embeddings", ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings", ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank", ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking", ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank", ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking", ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize", ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize", ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template", ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get("/lora-adapters", ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters", ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get("/slots", ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot", ex_wrapper(routes.post_slots));

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!ctx_http.start()) {
        try {
            ctx_http.stop();
        } catch (...) {
            SRV_ERR("%s: failed to stop HTTP server after start failure\n", __func__);
        }
        g_servers.erase(modelfilename);
        LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
        return;
    }

    // load the model
    LOG_INF("%s: checking whether model loaded\n", __func__);
    if (model_ctx.state != server_model_status::SERVER_MODEL_STATUS_LOADED) {
        try {
            ctx_http.stop();
        } catch (...) {
            SRV_ERR("%s: failed to stop HTTP server after start failure\n", __func__);
        }
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        g_servers.erase(modelfilename);
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return;
    }

    routes.update_meta(*model_ctx.server_ctx);
    ctx_http.is_ready.store(true);

    LOG_INF("%s: model loaded\n", __func__);
	
	// this call blocks the main thread until queue_tasks.terminate() is called
	if(model_ctx.state == server_model_status::SERVER_MODEL_STATUS_LOADED)
	{
		(*model_ctx.server_ctx).start_loop();
	}

    
}

void server_embedded_ggml_abort_callback_t(const char* error_message) {
#if defined(_DEBUG) || defined(DEBUG)
	fprintf(stderr, "%s", error_message);
#endif
}

void server_embedded_start(uint8_t numa_strategy, server_status_callback& callback) {
	if(callback){
		callback(server_embedded_status::SERVER_EMBEDDED_STATUS_STARTING);
	}
    try {
       auto& mem_info = get_memory_info();
        size_t total_mem = mem_info.total_physical/(1024*1024);
       if (total_mem > 4096) {
            g_modelManager.setMaxMemory(total_mem);
       }
    } catch (const std::exception & e) {
        LOG_WRN("%s: failed to get system memory info: %s\n", __func__, e.what());
    }
	
	ggml_set_abort_callback(server_embedded_ggml_abort_callback_t);

    common_init();
	
	// only print errors
	llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */) {
		if (level >= GGML_LOG_LEVEL_ERROR) {
			fprintf(stderr, "%s", text);
		}
		}, nullptr);

	// Based on tools/llama-bench/llama-bench.cpp
	// load dynamic backends
	ggml_backend_load_all();

    llama_backend_init();
	ggml_numa_strategy numa = ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED;
	switch(numa_strategy)
	{ 
		case 0: numa = ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED; break;
		case 1: numa = ggml_numa_strategy::GGML_NUMA_STRATEGY_DISTRIBUTE; break;
		case 2: numa = ggml_numa_strategy::GGML_NUMA_STRATEGY_ISOLATE; break;
		case 3: numa = ggml_numa_strategy::GGML_NUMA_STRATEGY_NUMACTL; break;
		case 4: numa = ggml_numa_strategy::GGML_NUMA_STRATEGY_MIRROR; break;
		default: numa = ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED; break;
	}
    llama_numa_init(numa);

    shutdown_handler = [](int) {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        std::vector<std::thread> to_join;
        for (auto & [name, ctx_http] : g_servers) {
            ctx_http->stop();
            to_join.push_back(std::move(ctx_http->thread));
        }
        for (auto & th : to_join) {
            if (th.joinable()) {
                th.join();
            }
        }
        auto models = g_modelManager.listModels();
        for (const auto & name : models) {
            SRV_INF("%s: unloading model %s\n", __func__, name.c_str());
            g_modelManager.unloadModel(name);
        }
        g_servers.clear();
        llama_backend_free();
    };
	
	// starting
	g_is_terminating.clear();
	if(callback){
		callback(server_embedded_status::SERVER_EMBEDDED_STATUS_STARTED);
	}
}

void server_embedded_stop(server_status_callback& callback){
	if (g_is_terminating.test_and_set()) {
		if(callback){
			callback(server_embedded_status::SERVER_EMBEDDED_STATUS_INVALID);
		}
        return;
    }
	if(callback){
		callback(server_embedded_status::SERVER_EMBEDDED_STATUS_STOPPING);
	}
    g_is_interrupted.store(true);
	if(shutdown_handler != nullptr){
		shutdown_handler(0);
	}
	if(callback){
		callback(server_embedded_status::SERVER_EMBEDDED_STATUS_STOPPED);
	}
}

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" || header_name == "transfer-encoding" ||
        header_name == "content-length" ||  // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static bool should_stop() {
    return g_is_interrupted.load();
}

llama_tokens server_embedded_tokenize_svc(std::string model, std::string text)
{
    ModelContext                    model_ctx  = g_modelManager.getModelContext(model);
    std::shared_ptr<server_context> server_ctx = model_ctx.server_ctx;
    server_context_meta &           meta               = server_ctx->get_meta();
    server_chat_params &            server_chat_params = meta.chat_params;
    llama_context* ctx = server_ctx->get_llama_context();
    return common_tokenize(ctx, text, false, false);
}

void server_embedded_add_model_status_listener(std::function<void(const std::string &, server_model_status, server_model_status)> listener)
{
	g_modelManager.addStateChangeListener(listener);
}

void server_embedded_rm_model_status_listeners(){
	g_modelManager.clearAllStateChangeListeners();
}

void server_embedded_submit(common_params_sampling sampling_params,
							std::string name,
                            std::vector<common_chat_msg>  messages,
                            std::vector<common_chat_tool> tools,
                            std::function<bool(std::string)> streaming_response_cb,
                            std::function<void(common_chat_msg_with_timings)> response_with_timings_cb) {
    ModelContext                    model_ctx  = g_modelManager.getModelContext(name);
    std::shared_ptr<server_context> server_ctx = model_ctx.server_ctx;
    // shared between reader and writer threads
    server_context_meta &           meta       = server_ctx->get_meta();
    server_chat_params &            server_chat_params = meta.chat_params;
    common_chat_templates_inputs inputs;
    {
        bool use_jinja               = !tools.empty() || server_chat_params.use_jinja;
        inputs.messages              = messages;
        inputs.tools                 = tools;
        inputs.use_jinja             = use_jinja;
        inputs.parallel_tool_calls   = false;
        inputs.add_generation_prompt = true;
		inputs.grammar 				 = sampling_params.grammar;
        auto chat_template_kwargs    = server_chat_params.chat_template_kwargs;
        for (const auto & [key, value] : chat_template_kwargs) {
            inputs.chat_template_kwargs[key] = value;
        }
		// for (const auto & [key, value] : meta.chat_template_caps) {
			// inputs.chat_template_kwargs[key] = value ? "true" : "false";
		// }
        inputs.tool_choice         = !tools.empty() ? common_chat_tool_choice::COMMON_CHAT_TOOL_CHOICE_AUTO :
                                                      common_chat_tool_choice::COMMON_CHAT_TOOL_CHOICE_NONE;

        auto enable_thinking_kwarg = json_value(chat_template_kwargs, "enable_thinking", server_chat_params.enable_thinking ? std::string("true") : std::string("false"));
        if (server_chat_params.reasoning_format != common_reasoning_format::COMMON_REASONING_FORMAT_NONE || enable_thinking_kwarg == "true") {
            inputs.enable_thinking = use_jinja;
        } else if (enable_thinking_kwarg == "false") {
            inputs.enable_thinking = false;
        } else {
            inputs.enable_thinking = server_chat_params.enable_thinking;
        }
		inputs.reasoning_format = inputs.enable_thinking ? server_chat_params.reasoning_format !=
                                                       common_reasoning_format::COMMON_REASONING_FORMAT_NONE ?
                                                        server_chat_params.reasoning_format :
                                                        common_reasoning_format::COMMON_REASONING_FORMAT_AUTO
                                                 : server_chat_params.reasoning_format;
    }
    common_chat_params chat_params = common_chat_templates_apply(server_chat_params.tmpls.get(), inputs);
    
    auto & generate_completion = [&server_ctx, &streaming_response_cb , &response_with_timings_cb,
									sampling_params, inputs, chat_params]() {
        result_timings out_timings;
        server_response_reader rd = server_ctx->get_response_reader();
        server_task            task = server_task(SERVER_TASK_TYPE_COMPLETION);
		task_params            server_task_params;
		server_task_params.sampling          = sampling_params;
		server_task_params.stream            = true;  // make sure we always use streaming mode
		server_task_params.timings_per_token = true;  // in order to get timings even when we cancel mid-way
		server_task_params.chat_parser_params.reasoning_format     = inputs.reasoning_format;
		server_task_params.chat_parser_params.thinking_forced_open = chat_params.thinking_forced_open;
		server_task_params.chat_parser_params.format               = chat_params.format;
		server_task_params.chat_parser_params.reasoning_in_content =
			server_task_params.stream &&
			inputs.reasoning_format != common_reasoning_format::COMMON_REASONING_FORMAT_NONE;
		server_task_params.chat_parser_params.parse_tool_calls = inputs.tool_choice != common_chat_tool_choice::COMMON_CHAT_TOOL_CHOICE_NONE;
        {
            std::vector<raw_buffer> input_files;
            // TODO: reduce some copies here in the future
            task.id                      = rd.get_new_id();
            task.index                   = 0;
            task.params                  = server_task_params;            // copy
			// OT USING  MTMD
            // task.cli_prompt              = chat_params.prompt;  // copy
            // task.cli_files               = input_files;         // copy
            // task.cli                     = true;

            // chat template settings
            // task.params.chat_parser_params                  = common_chat_parser_params(chat_params);
            // task.params.chat_parser_params.reasoning_format = server_chat_params.reasoning_format;
            if (!chat_params.parser.empty()) {
                task.params.chat_parser_params.parser.load(chat_params.parser);
            }

        }
        rd.post_task({ std::move(task) });
        // wait for first result
        server_task_result_ptr result = rd.next(should_stop);
        std::string            curr_content, reasoning_content;
        bool                   is_thinking = false;

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    LOG_ERR("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    LOG_ERR("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = std::move(res_partial->timings);
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (is_thinking) {
                            is_thinking = false;
                        }
                        if (streaming_response_cb) {
                            if(streaming_response_cb(diff.content_delta)){
								break;
							}
                        }
                        curr_content += diff.content_delta;
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        is_thinking = true;
                        if (streaming_response_cb) {
                            if(streaming_response_cb(diff.reasoning_content_delta)){
								break;
							}
                        }
                        reasoning_content += diff.reasoning_content_delta;
                    }
                }
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                break;
            }
            result = rd.next(should_stop);
        }
       
        // server_response_reader automatically cancels pending tasks upon destruction
        g_is_interrupted.store(false);
        if (response_with_timings_cb) {
            common_chat_msg message = common_chat_parse(curr_content, g_is_interrupted.load(), task.params.chat_parser_params);
            message.reasoning_content = reasoning_content;
            common_chat_msg_with_timings msg_with_timings(message, out_timings);
            response_with_timings_cb(msg_with_timings);
        }
    };
    generate_completion();
}

server_core_proxy::server_core_proxy(const std::string &                        method,
                                     const std::string &                        path,
									 const std::string &						query_string,
                                     const std::map<std::string, std::string> & headers,
                                     const std::string &                        body,
                                     const std::function<bool()>                should_stop,
                                     int32_t                                    timeout_read,
                                     int32_t                                    timeout_write) {
    std::string name = json_value(body, "model", std::string());
	json jmessages = json_value(body, "messages", json::array());
    json jtools = json_value(body, "tools", json());
	
	auto& messages = common_chat_msgs_parse_oaicompat(jmessages);
	auto& tools = common_chat_tools_parse_oaicompat(jtools);
	ModelContext model_ctx = g_modelManager.getModelContext(name);
	
	if (model_ctx.state != SERVER_MODEL_STATUS_LOADED) {
        json err_msg;
		std::string message("Model ");
		message.append(name).append("not loaded");
		err_msg["status"]= -1;
		err_msg["message"] = message;
		this->data = err_msg.dump();
		return;
    }
	
	server_core_context* core_ctx = g_servers[name];
    std::shared_ptr<MemoryDuplexStream> conn = core_ctx->srv->create_connection();
	
	// wire up the receive end of the pipe
    this->next = [&conn](std::string & out) -> bool {
        return conn->recv_from_server(out); // false if EOF or pipe broken
    };

    auto check = [&]() -> bool {
        return should_stop();
    };
	
	auto streaming_response_cb = [&](std::string out) -> bool {
        int err = conn->write(out.c_str(), strlen(out.c_str()));
        return check() || err < 0;
    };

    auto response_with_timings_cb = [&](common_chat_msg_with_timings out) {
        std::vector<common_chat_msg> msgs;
        msgs.push_back(out.message);
        auto resp  = common_chat_msgs_to_json_oaicompat(msgs, false);
        this->data = resp.dump();
    };
	
	common_params_sampling sampling;
	this->thread = std::thread([&]() {
            server_embedded_submit(sampling, name, messages, tools, streaming_response_cb, response_with_timings_cb);
		});
	this->thread.detach();
	
	
}

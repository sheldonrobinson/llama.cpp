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
static std::function<void(int)> shutdown_handler;
static std::atomic_flag g_is_terminating = ATOMIC_FLAG_INIT;
static std::atomic<bool> g_is_interrupted = false;
static std::unordered_map<std::string, std::shared_ptr<server_core_context>> g_servers;

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

    auto & result =
        g_servers.emplace(std::make_pair(modelfilename, std::move(std::make_shared<server_core_context>())));

    if (!result.second) {
        return;
    }

    auto & ctx_http               = result.first->second;
    if (!ctx_http->init(params)) {
        g_servers.erase(modelfilename);
        return;
    }

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!ctx_http->start()) {
        try {
            ctx_http->stop();
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
            ctx_http->stop();
        } catch (...) {
            SRV_ERR("%s: failed to stop HTTP server after start failure\n", __func__);
        }
        if (ctx_http->thread.joinable()) {
            ctx_http->thread.join();
        }
        g_servers.erase(modelfilename);
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return;
    }

   ctx_http->is_ready.store(true);

    LOG_INF("%s: model loaded\n", __func__);
   
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

bool server_embedded_submit(common_params_sampling sampling_params,
							std::string name,
                            std::vector<common_chat_msg>  messages,
                            std::vector<common_chat_tool> tools,
                            std::function<bool(std::string)> streaming_response_cb,
                            std::function<void(common_chat_msg)> response_cb) {
    ModelContext                    model_ctx  = g_modelManager.getModelContext(name);
	
	if(model_ctx.state != server_model_status::SERVER_MODEL_STATUS_LOADED)
	{
		return false;
	}
	
    std::shared_ptr<server_context> server_ctx = model_ctx.server_ctx;
    server_context_meta             meta               = server_ctx->get_meta();
    server_chat_params &            server_chat_params = meta.chat_params;
    common_chat_templates_inputs    inputs;

        
        
        {
            bool use_jinja               = !tools.empty() || server_chat_params.use_jinja;
            inputs.messages              = messages;
            inputs.tools                 = tools;
            inputs.use_jinja             = use_jinja;
            inputs.parallel_tool_calls   = false;
            inputs.add_generation_prompt = true;
            inputs.grammar               = sampling_params.grammar;
            inputs.tool_choice           = !tools.empty() || server_chat_params.use_jinja ?
                                               common_chat_tool_choice::COMMON_CHAT_TOOL_CHOICE_AUTO :
                                               common_chat_tool_choice::COMMON_CHAT_TOOL_CHOICE_NONE;
            auto chat_template_kwargs    = server_chat_params.chat_template_kwargs;
            for (const auto & [key, value] : chat_template_kwargs) {
                inputs.chat_template_kwargs[key] = value;
            }

            auto enable_thinking_kwarg =
                json_value(chat_template_kwargs, "enable_thinking",
                           server_chat_params.enable_thinking ? std::string("true") : std::string("false"));
            if (server_chat_params.reasoning_format != common_reasoning_format::COMMON_REASONING_FORMAT_NONE ||
                server_chat_params.enable_thinking || enable_thinking_kwarg == "true") {
                inputs.enable_thinking = true;
            } else if (enable_thinking_kwarg == "false") {
                inputs.enable_thinking = false;
            } else {
                inputs.enable_thinking = server_chat_params.enable_thinking;
            }
            inputs.reasoning_format =
                inputs.enable_thinking ?
                    server_chat_params.reasoning_format != common_reasoning_format::COMMON_REASONING_FORMAT_NONE ?
                    server_chat_params.reasoning_format :
                    common_reasoning_format::COMMON_REASONING_FORMAT_AUTO :
                    server_chat_params.reasoning_format;
        }

    // Apply chat template to the list of messages
    common_chat_params params = common_chat_templates_apply(server_chat_params.tmpls.get(), inputs);

    embedded_context embedded_ctx(
        params,
        sampling_params,
        messages, tools,
        streaming_response_cb,
        response_cb,
        should_stop
    );
	// this call blocks the main thread until queue_tasks.terminate() is called
	
	std::thread inference_thread([&server_ctx]() { server_ctx->start_loop(); });

    result_timings timings;
    std::string    assistant_content = embedded_ctx.generate_completion(
        server_ctx->get_response_reader(),
        timings);
	if(inference_thread.joinable()){
		inference_thread.join();
	}
	return true;
}

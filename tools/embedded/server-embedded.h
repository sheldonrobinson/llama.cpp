#include "common.h"
#include "preset.h"
#include "server-common.h"
#include "server-core.h"

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <set>

#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include "httplib.h" // cpp-httplib header
#include "concurrentqueue.h" // moodycamel concurrentqueue

#ifdef __cplusplus
	#include <cstdint>
	#include <cstdbool>
#else // __cplusplus - Objective-C or other C platform
	#include <stdint.h>
	#include <stdbool.h>
#endif


#ifdef __cplusplus
extern "C"
{
#endif

/**
 * state diagram:
 *
 * UNLOADED ──► LOADING ──► LOADED
 *  ▲            │            │
 *  └───failed───┘            │
 *  ▲                         │
 *  └────────unloaded─────────┘
 */

enum server_model_status : uint8_t {
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED
};

static server_model_status server_model_status_from_string(const std::string & status_str) {
    if (status_str == "unloaded") {
        return SERVER_MODEL_STATUS_UNLOADED;
    }
    if (status_str == "loading") {
        return SERVER_MODEL_STATUS_LOADING;
    }
    if (status_str == "loaded") {
        return SERVER_MODEL_STATUS_LOADED;
    }
    throw std::runtime_error("invalid server model status");
}

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED: return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:  return "loading";
        case SERVER_MODEL_STATUS_LOADED:   return "loaded";
        default:                           return "unknown";
    }
}

typedef enum server_embedded_status : uint8_t {
    // TODO: also add downloading state when the logic is added
    SERVER_EMBEDDED_STATUS_STARTING,
    SERVER_EMBEDDED_STATUS_STARTED,
    SERVER_EMBEDDED_STATUS_BUSY,
	SERVER_EMBEDDED_STATUS_IDLE,
	SERVER_EMBEDDED_STATUS_STOPPING,
	SERVER_EMBEDDED_STATUS_STOPPED,
	SERVER_EMBEDDED_STATUS_NOT_FOUND
} server_embedded_status_t;

typedef void (*server_status_callback)(server_embedded_status_t, size_t);

#ifdef __cplusplus
}
#endif

typedef struct server_model_meta {
    common_preset preset;
    std::string name;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout = 0; // seconds to wait before force-killing the model instance during shutdown

    bool is_active() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
} server_model_meta_t;

typedef struct server_models {
private:
    struct instance_t {
        std::thread th;
        server_model_meta_t meta;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // for stopping models
    std::condition_variable cv_stop;
    std::set<std::string> stopping_models;

    common_preset_context ctx_preset;
	
    common_params base_params;
	std::string bin_path;

    void update_meta(const std::string & name, const server_model_meta_t & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta_t && meta);

public:
    server_models(const common_params & params);

    void load_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta_t> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta_t> get_all_meta();

    // load and unload model instances
    // these functions are thread-safe
    void load(const std::string & name);
    void unload(const std::string & name);
    void unload_all();

    // update the status of a model instance (thread-safe)
    void update_status(const std::string & name, server_model_status status, int exit_code);

    // wait until the model instance is fully loaded (thread-safe)
    // return when the model is loaded or failed to load
    void wait_until_loaded(const std::string & name);

    // load the model if not loaded, otherwise do nothing (thread-safe)
    // return false if model is already loaded; return true otherwise (meta may need to be refreshed)
    bool ensure_model_loaded(const std::string & name);
} server_models_t;

struct server_models_routes {
    common_params params;
    server_models_t models;
    server_models_routes(const common_params & params)
            : params(params), models(params) {
        init_routes();
    }

    void init_routes();
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    server_core_context::handler_t get_router_props;
    server_core_context::handler_t proxy_call;
    server_core_context::handler_t get_router_models;
    server_core_context::handler_t post_router_models_load;
    server_core_context::handler_t post_router_models_unload;
} server_models_t;

LLAMA_API void server_embedded_start(const common_params& params, server_status_callback* callback);

LLAMA_API void server_embedded_stop(size_t tid);
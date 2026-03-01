#pragma once

#include "common.h"
#include "preset.h"
#include "server-common.h"
#include "server-core.h"
#include "server-task.h"
#include "server-queue.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#ifdef __cplusplus
#    include <cstdbool>
#    include <cstdint>
#else  // __cplusplus - Objective-C or other C platform
#    include <stdbool.h>
#    include <stdint.h>
#endif

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_EMBEDDED_API __declspec(dllexport)
#        else
#            define LLAMA_EMBEDDED_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_EMBEDDED_API __attribute__((visibility("default")))
#    endif
#else
#    define LLAMA_EMBEDDED_API
#endif

#ifdef __cplusplus
extern "C" {
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

enum server_model_status {
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED,
    SERVER_MODEL_STATUS_INVALID
};

typedef enum server_embedded_status {
    // TODO: also add downloading state when the logic is added
    SERVER_EMBEDDED_STATUS_STARTING,
    SERVER_EMBEDDED_STATUS_STARTED,
    SERVER_EMBEDDED_STATUS_STOPPING,
    SERVER_EMBEDDED_STATUS_STOPPED,
    SERVER_EMBEDDED_STATUS_INVALID
} server_embedded_status_t;

typedef void (*server_status_callback)(server_embedded_status_t);

#ifdef __cplusplus
}
#endif

static server_model_status server_model_status_from_string(const std::string & status_str) {
    if (status_str == "unloaded") {
        return server_model_status::SERVER_MODEL_STATUS_UNLOADED;
    }
    if (status_str == "loading") {
        return server_model_status::SERVER_MODEL_STATUS_LOADING;
    }
    if (status_str == "loaded") {
        return server_model_status::SERVER_MODEL_STATUS_LOADED;
    }
    return server_model_status::SERVER_MODEL_STATUS_INVALID;
}

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED:
            return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:
            return "loading";
        case SERVER_MODEL_STATUS_LOADED:
            return "loaded";
        default:
            return "invalid";
    }
}

struct server_model_meta {
    common_preset            preset;
    std::string              name;
    std::set<std::string>    aliases;        // additional names that resolve to this model
    std::set<std::string>    tags;           // informational tags, not used for routing
    server_model_status      status    = server_model_status::SERVER_MODEL_STATUS_UNLOADED;
    int64_t                  last_used = 0;  // for LRU unloading
    std::vector<std::string> args;           // args passed to the model instance, will be populated by render_args()
    int                      exit_code = 0;  // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout                   = 0;  // seconds to wait before force-killing the model instance during shutdown

    bool is_active() const { return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING; }

    bool is_failed() const {
        return (status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0) || status == SERVER_MODEL_STATUS_INVALID;
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
};

struct server_models {
  private:
    struct instance_t {
        std::thread       th;
        server_model_meta meta;
    };

    std::mutex                        mutex;
    std::condition_variable           cv;
    std::map<std::string, instance_t> mapping;

    // for stopping models
    std::condition_variable cv_stop;
    std::set<std::string>   stopping_models;

    common_preset_context ctx_preset;

    common_params base_params;
    std::string   bin_path;

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

  public:
    server_models(const common_params & params);

    void load_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

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

    // proxy an HTTP request to the model instance
    server_core_res_ptr proxy_request(const server_core_req & req,
                                      const std::string &     method,
                                      const std::string &     name,
                                      bool                    update_last_used);

    // load the model if not loaded, otherwise do nothing (thread-safe)
    // return false if model is already loaded; return true otherwise (meta may need to be refreshed)
    bool ensure_model_loaded(const std::string & name);
};

struct embedded_context {
    common_chat_params                                chat_params;
    std::vector<common_chat_msg>     messages;
    std::vector<common_chat_tool>   tools;
    std::vector<raw_buffer>                           input_files;
    task_params                                       server_task_params;
    std::function<bool(std::string)>                  streaming_response_cb    = nullptr;
    std::function<void(common_chat_msg)> response_with_timings_cb = nullptr;
    std::function<bool()>              should_stop = nullptr;

    embedded_context(common_chat_params    params,
                     common_params_sampling          sampling,
                    std::vector<common_chat_msg>      msgs,
                    std::vector<common_chat_tool> toolcalls,
                    std::function<bool(std::string)>&  streaming_cb,
                    std::function<void(common_chat_msg)>& response_cb,
                     std::function<bool()>&                  should_function
    ) {
        chat_params                          = params;  
                    server_task_params.sampling  = sampling;
                    messages                     = msgs;
                    tools                        = toolcalls;
                    streaming_response_cb        = streaming_cb;
                    response_with_timings_cb     = response_cb;
					should_stop					 = stop_function;
                    server_task_params.stream  = true;  // make sure we always use streaming mode
                    server_task_params.timings_per_token = true;  // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress
    }

    std::string generate_completion(server_response_reader rd, result_timings & out_timings) {
        {
            server_task_params.chat_parser_params.thinking_forced_open = chat_params.thinking_forced_open;
            server_task_params.chat_parser_params.format               = chat_params.format;
            server_task_params.chat_parser_params.reasoning_in_content = 
                server_task_params.stream && server_task_params.chat_parser_params.reasoning_format !=
                                                 common_reasoning_format::COMMON_REASONING_FORMAT_NONE;
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id          = rd.get_new_id();
            task.index       = 0;
            task.params      = server_task_params;  // copy
            task.cli_prompt  = chat_params.prompt;  // copy
            task.cli_files   = input_files;         // copy
            task.cli         = true;

            // chat template settings
            task.params.chat_parser_params                  = common_chat_parser_params(chat_params);
            task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            if (!chat_params.parser.empty()) {
                task.params.chat_parser_params.parser.load(chat_params.parser);
            }

            rd.post_task({ std::move(task) });
        }

        // wait for first result
        server_task_result_ptr result = rd.next(should_stop);
        std::string curr_content, reasoning_content;
        bool        is_thinking = false;
        bool        is_partial_result =
            false;  // whether we have received any partial result (used to determine whether to call the callback for the first time)
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
                            if (streaming_response_cb(diff.content_delta)) {
                                break;
                            }
                        }
                        curr_content += diff.content_delta;
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        is_thinking = true;
                        if (streaming_response_cb) {
                            if (streaming_response_cb(diff.reasoning_content_delta)) {
                                break;
                            }
                        }
                        reasoning_content += diff.reasoning_content_delta;
                    }
                }
                is_partial_result = true;
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                is_partial_result = false;
                break;
            }
            result = rd.next(should_stop);
        }
        common_chat_msg message =
            common_chat_parse(curr_content, is_partial_result, server_task_params.chat_parser_params);
        message.reasoning_content = reasoning_content;
        if (response_with_timings_cb) {
            response_with_timings_cb(message);
        }
        // server_response_reader automatically cancels pending tasks upon destruction
        return message.to_json_oaicompat().dump();
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return mtmd_default_marker();
        } else {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return content;
        }
    }
};

struct common_chat_msg_with_timings {
    common_chat_msg message;
    result_timings  timings;
    common_chat_msg_with_timings()  = default;
    ~common_chat_msg_with_timings() = default;

    common_chat_msg_with_timings(common_chat_msg msg, result_timings metrics) :
        message(std::move(msg)),
        timings(std::move(metrics)) {}

    common_chat_msg_with_timings(common_chat_msg_with_timings & other) :
        message(other.message),
        timings(other.timings) {}
};

LLAMA_EMBEDDED_API void server_embedded_add_model_status_listener(
    std::function<void(const std::string &, server_model_status, server_model_status)> listener);

LLAMA_EMBEDDED_API void server_embedded_rm_model_status_listeners();

LLAMA_EMBEDDED_API void server_embedded_inference_svc(common_params args);

LLAMA_EMBEDDED_API llama_tokens server_embedded_tokenize_svc(std::string model, std::string text);

LLAMA_EMBEDDED_API void server_embedded_start(uint8_t numa_strategy, server_status_callback & callback);

LLAMA_EMBEDDED_API void server_embedded_stop(server_status_callback & callback);

LLAMA_EMBEDDED_API bool server_embedded_submit(
    common_params_sampling                            sampling_params,
    std::string                                       model,
    std::vector<common_chat_msg>                      messages,
    std::vector<common_chat_tool>                     tools,
    std::function<bool(std::string)>                  streaming_response_cb,
    std::function<void(common_chat_msg)> response_with_timings_cb);

LLAMA_EMBEDDED_API std::string server_embedded_model_list();

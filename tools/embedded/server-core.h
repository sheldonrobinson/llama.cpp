#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>

struct common_params;

enum server_core_status : uint8_t {
	SERVER_CORE_STATUS_SUCCESS,
	SERVER_CORE_STATUS_FAILURE
}

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
typedef struct server_core_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = SERVER_CORE_STATUS_SUCCESS;
    std::string data;
    std::map<std::string, std::string> metadata;

    // TODO: move this to a virtual function once we have proper polymorphism support
    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    virtual ~server_core_res() = default;
} server_core_res_t;

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_core_res_ptr = std::unique_ptr<server_core_res>;

typedef struct server_core_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> metadata; // reserved for future use
    std::string path; // reserved for future use
    std::string body;
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }
} server_core_req_t;


struct server_core_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    server_core_context();
    ~server_core_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_core_res_ptr(const server_core_req & req)>;

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;
};




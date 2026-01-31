#include "common.h"
#include "server-core.h"
#include "server-common.h"

#include "uv-memory-server.hpp"
#include <functional>
#include <string>
#include <thread>


//
// HTTP implementation using cpp-httplib, any multithread middleware allowed
//
server_core_context::~server_core_context() = default;

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health") {
        return;
    }

    // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff, think about wrapping it in try-catch

    SRV_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);

    SRV_DBG("request:  %s\n", req.body.c_str());
    SRV_DBG("response: %s\n", res.body.c_str());
}

bool server_core_context::init(const common_params & params) {

    srv.reset(new UVMemoryServer(4,50,8));
	
	srv->set_default_headers({{"Server", "llama.cpp"}});
    srv->set_logger(log_server_request);
    srv->set_exception_handler([](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        // this is fail-safe; exceptions should already handled by `ex_wrapper`

        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        res.status = 500;
        res.set_content(message, "text/plain");
        LOG_ERR("got exception: %s\n", message.c_str());
    });

    srv->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res.set_content(
                safe_json_to_str(json {
                    {"error", {
                        {"message", "File Not Found"},
                        {"type", "not_found_error"},
                        {"code", 404}
                    }}
                }),
                "application/json; charset=utf-8"
            );
        }
        // for other error codes, we skip processing here because it's already done by res->error()
    });

    // set timeouts and change hostname and port
    srv->set_read_timeout (params.timeout_read);
    srv->set_write_timeout(params.timeout_write);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        std::string substr = key.substr(std::max((int)(key.length() - 4), 0));
        LOG_INF("%s: api_keys: ****%s\n", __func__, substr.c_str());
    } else if (params.api_keys.size() > 1) {
        LOG_INF("%s: api_keys: %zu keys loaded\n", __func__, params.api_keys.size());
    }

    //
    // Middlewares
    //

    auto middleware_validate_api_key = [api_keys = params.api_keys](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = {
            "/health",
            "/v1/health",
            "/models",
            "/v1/models",
            "/api/tags"
        };

        // If API key is not set, skip validation
        if (api_keys.empty()) {
            return true;
        }

        // If path is public or is static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
            return true;
        }

        // Check for API key in the Authorization header
        std::string req_api_key = req.get_header_value("Authorization");
        if (req_api_key.empty()) {
            // retry with anthropic header
            req_api_key = req.get_header_value("X-Api-Key");
        }

        // remove the "Bearer " prefix if needed
        std::string prefix = "Bearer ";
        if (req_api_key.substr(0, prefix.size()) == prefix) {
            req_api_key = req_api_key.substr(prefix.size());
        }

        // validate the API key
        if (std::find(api_keys.begin(), api_keys.end(), req_api_key) != api_keys.end()) {
            return true; // API key is valid
        }

        // API key is invalid or not provided
        res.status = 401;
        res.set_content(
            safe_json_to_str(json {
                {"error", {
                    {"message", "Invalid API Key"},
                    {"type", "authentication_error"},
                    {"code", 401}
                }}
            }),
            "application/json; charset=utf-8"
        );

        LOG_WRN("Unauthorized: Invalid API Key\n");

        return false;
    };

    auto middleware_server_state = [this](const httplib::Request & req, httplib::Response & res) {
        bool ready = is_ready.load();
        if (!ready) {
			// no endpoints is allowed to be accessed when the server is not ready
			// this is to prevent any data races or inconsistent states
			res.status = 503;
			res.set_content(
				safe_json_to_str(json {
					{"error", {
						{"message", "Loading model"},
						{"type", "unavailable_error"},
						{"code", 503}
					}}
				}),
				"application/json; charset=utf-8"
			);
            return false;
        }
        return true;
    };

    // register server middlewares
    srv->set_pre_routing_handler([middleware_validate_api_key, middleware_server_state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    int n_threads_http = params.n_threads_http;
    if (n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    LOG_INF("%s: using %d threads for HTTP server\n", __func__, n_threads_http);
    srv->new_task_queue = [n_threads_http] { 
		return new httplib::ThreadPool(n_threads_http); 
	};
	
	return true;
}
bool server_core_context::start() {
    // Bind and listen

    // run the HTTP server in a thread
    thread = std::thread([this]() { srv->listen_with_uv(); });
    srv->wait_until_ready();
    return true;
}

void server_core_context::stop() const {
    if (srv) {
        srv->stop_with_uv();
    }
}

static void set_metadata(httplib::Response & res, const std::map<std::string, std::string> & headers) {
    for (const auto & [key, value] : headers) {
        res.set_header(key, value);
    }
}

static std::map<std::string, std::string> get_params(const httplib::Request & req) {
    std::map<std::string, std::string> params;
    for (const auto & [key, value] : req.params) {
        params[key] = value;
    }
    for (const auto & [key, value] : req.path_params) {
        params[key] = value;
    }
    return params;
}

static std::map<std::string, std::string> get_headers(const httplib::Request & req) {
    std::map<std::string, std::string> headers;
    for (const auto & [key, value] : req.headers) {
        headers[key] = value;
    }
    return headers;
}

// using unique_ptr for request to allow safe capturing in lambdas
using server_core_req_ptr = std::unique_ptr<server_core_req>;

static void process_handler_response(server_core_req_ptr && request, server_core_res_ptr & response, httplib::Response & res) {
    if (response->is_stream()) {
        res.status = response->status;
        set_metadata(res, response->metadata);
        std::string content_type = response->content_type;
        // convert to shared_ptr as both chunked_content_provider() and on_complete() need to use it
        std::shared_ptr<server_core_req> q_ptr = std::move(request);
        std::shared_ptr<server_core_res> r_ptr = std::move(response);
        const auto chunked_content_provider = [response = r_ptr](size_t, httplib::DataSink & sink) -> bool {
            std::string chunk;
            bool has_next = response->next(chunk);
            if (!chunk.empty()) {
                // TODO: maybe handle sink.write unsuccessful? for now, we rely on is_connection_closed()
                sink.write(chunk.data(), chunk.size());
                SRV_DBG("http: streamed chunk: %s\n", chunk.c_str());
            }
            if (!has_next) {
                sink.done();
                SRV_DBG("%s", "http: stream ended\n");
            }
            return has_next;
        };
        const auto on_complete = [request = q_ptr, response = r_ptr](bool) mutable {
            response.reset(); // trigger the destruction of the response object
            request.reset();  // trigger the destruction of the request object
        };
        res.set_chunked_content_provider(content_type, chunked_content_provider, on_complete);
    } else {
        res.status = response->status == 200 ? SERVER_CORE_STATUS_SUCCESS : SERVER_CORE_STATUS_FAILURE;
        set_metadata(res, response->metadata);
        res.set_content(response->data, response->content_type);
    }
}

void server_core_context::get(const std::string & path, const server_core_context::handler_t & handler) const {
    srv->Get(path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_core_req_ptr request = std::make_unique<server_core_req>(server_core_req{
            get_params(req),
            get_headers(req),
            req.path,
            req.body,
            req.is_connection_closed
        });
        server_core_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}

void server_core_context::post(const std::string & path, const server_core_context::handler_t & handler) const {
    srv->Post(path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_core_req_ptr request = std::make_unique<server_core_req>(server_core_req{
            get_params(req),
            get_headers(req),
            req.path,
            req.body,
            req.is_connection_closed
        });
        server_core_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}

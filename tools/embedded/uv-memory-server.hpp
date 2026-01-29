#pragma once

#include <cpp-httplib/httplib.h>
#include <uv.h>
#include <pthreadpool.h>
#include <thread>
#include <atomic>
#include <memory>
#include <iostream>
#include <sstream>
#include <functional>
#include <string>
#include <unordered_set>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>

// ====== Simple Logger ======
#ifndef LOG_INFO
#define LOG_INFO(msg)  std::cout << "[INFO] " << msg << "\n"
#endif
#ifndef LOG_ERROR
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << "\n"
#endif

// ====== Metrics Struct ======
struct ServerMetrics {
    std::atomic<size_t> active_connections{0};
    // std::atomic<size_t> total_messages_received{0};
    // std::atomic<size_t> total_messages_sent{0};
    // std::atomic<size_t> errors{0};
};

// ====== Lossless Blocking In-Memory Stream ======
class MemoryDuplexStream : public httplib::Stream {
public:
    MemoryDuplexStream(uv_loop_t* loop, size_t max_queue_size, ServerMetrics& metrics)
        : loop_(loop), max_queue_size_(max_queue_size), closed_(false), metrics_(metrics) {
        uv_async_init(loop_, &async_handle_, [](uv_async_t* handle) {
            auto* self = static_cast<MemoryDuplexStream*>(handle->data);
            self->deliver_messages();
        });
        async_handle_.data = this;
    }

    ~MemoryDuplexStream() {
        close();
        uv_close(reinterpret_cast<uv_handle_t*>(&async_handle_), nullptr);
    }

    // ====== Server reads from client ======
    ssize_t read(char* ptr, size_t size) override {
        std::unique_lock<std::mutex> lock(c2s_mutex_);
        c2s_cv_not_empty_.wait(lock, [this] {
            return !client_to_server_queue_.empty() || closed_;
        });

        if (closed_ && client_to_server_queue_.empty()) return 0;

        std::string msg = std::move(client_to_server_queue_.front());
        client_to_server_queue_.pop();
        c2s_cv_not_full_.notify_one();

        size_t len = std::min(size, msg.size());
        memcpy(ptr, msg.data(), len);
        return static_cast<ssize_t>(len);
    }

    // ====== Server writes to client ======
    ssize_t write(const char* ptr, size_t size) override {
        if (closed_) return -1;

        std::unique_lock<std::mutex> lock(s2c_mutex_);
        s2c_cv_not_full_.wait(lock, [this] {
            return server_to_client_queue_.size() < max_queue_size_ || closed_;
        });

        if (closed_) return -1;

        server_to_client_queue_.push(std::string(ptr, size));
        // metrics_.total_messages_sent++;
        s2c_cv_not_empty_.notify_one();
        return static_cast<ssize_t>(size);
    }

    void get_remote_ip_and_port(std::string &ip, int &port) const override {
        ip = "0.0.0.0";
        port = 0;
    }

    // ====== Client sends to server ======
    void send_to_server(const std::string& data) {
        if (closed_) return;

        std::unique_lock<std::mutex> lock(c2s_mutex_);
        c2s_cv_not_full_.wait(lock, [this] {
            return client_to_server_queue_.size() < max_queue_size_ || closed_;
        });

        if (closed_) return;

        client_to_server_queue_.push(data);
        c2s_cv_not_empty_.notify_one();
        uv_async_send(&async_handle_);
    }

    // ====== Client reads from server ======
    bool recv_from_server(std::string& out) {
        std::unique_lock<std::mutex> lock(s2c_mutex_);
        s2c_cv_not_empty_.wait(lock, [this] {
            return !server_to_client_queue_.empty() || closed_;
        });

        if (server_to_client_queue_.empty()) return false;

        out = std::move(server_to_client_queue_.front());
        server_to_client_queue_.pop();
        s2c_cv_not_full_.notify_one();
        return true;
    }

    void set_message_handler(std::function<void(const std::string&)> cb) {
        std::lock_guard<std::mutex> lock(handler_mutex_);
        message_handler_ = std::move(cb);
    }

    void close() {
        closed_ = true;

        // Wake up all waiting threads
        c2s_cv_not_empty_.notify_all();
        c2s_cv_not_full_.notify_all();
        s2c_cv_not_empty_.notify_all();
        s2c_cv_not_full_.notify_all();

        std::lock_guard<std::mutex> lock(handler_mutex_);
        message_handler_ = nullptr;
    }

    bool is_closed() const { return closed_; }

private:
    void deliver_messages() {
        std::queue<std::string> local_queue;

        {
            std::unique_lock<std::mutex> lock(c2s_mutex_);
            while (!client_to_server_queue_.empty()) {
                local_queue.push(std::move(client_to_server_queue_.front()));
                client_to_server_queue_.pop();
                c2s_cv_not_full_.notify_one();
            }
        }

        while (!local_queue.empty()) {
            // metrics_.total_messages_received++;
            std::function<void(const std::string&)> handler_copy;
            {
                std::lock_guard<std::mutex> lock(handler_mutex_);
                handler_copy = message_handler_;
            }
            if (handler_copy) {
                try {
                    handler_copy(local_queue.front());
                } catch (const std::exception& e) {
                    // metrics_.errors++;
                    LOG_ERROR("Handler exception: " << e.what());
                }
            }
            local_queue.pop();
        }
    }

    uv_loop_t* loop_;
    uv_async_t async_handle_;

    // Lossless queues
    std::queue<std::string> client_to_server_queue_;
    std::queue<std::string> server_to_client_queue_;

    // Synchronization
    std::mutex c2s_mutex_;
    std::condition_variable c2s_cv_not_empty_;
    std::condition_variable c2s_cv_not_full_;

    std::mutex s2c_mutex_;
    std::condition_variable s2c_cv_not_empty_;
    std::condition_variable s2c_cv_not_full_;

    std::function<void(const std::string&)> message_handler_;
    std::mutex handler_mutex_;

    size_t max_queue_size_;
    std::atomic<bool> closed_;
    ServerMetrics& metrics_;
};

// ====== UVMemoryServer ======
class UVMemoryServer : public httplib::Server {
public:
    using StreamPtr = std::shared_ptr<MemoryDuplexStream>;

    UVMemoryServer(size_t thread_count = 4, size_t max_queue_size = 100, size_t batch_size = 8)
        : max_queue_size_(max_queue_size), batch_size_(batch_size) {
        loop_ = uv_loop_new();
		// Alright — here’s the rest of the uv_memory_server.hpp starting from the UVMemoryServer definition so you have the complete drop‑in replacement with the lossless blocking wait / flow control mechanism integrated.
        uv_async_init(loop_, &async_handle_, [](uv_async_t* handle) {
            auto* self = static_cast<UVMemoryServer*>(handle->data);
            self->process_once();
        });
        async_handle_.data = this;

        pool_ = pthreadpool_create(thread_count);
        if (!pool_) {
            throw std::runtime_error("Failed to create pthreadpool");
        }
    }

    ~UVMemoryServer() {
        stop_with_uv();
        if (pool_) pthreadpool_destroy(pool_);
        uv_loop_close(loop_);
        uv_loop_delete(loop_);
    }

    // Create a new in-memory connection (client simulation)
    StreamPtr create_connection() {
        auto stream = std::make_shared<MemoryDuplexStream>(loop_, max_queue_size_, metrics_);
        {
            std::lock_guard<std::mutex> lock(conn_mutex_);
            connections_.insert(stream);
            metrics_.active_connections++;
        }
        conn_queue_.push_back(stream);
        uv_async_send(&async_handle_);
        return stream;
    }

    // Start the libuv event loop
    bool listen_with_uv() {
        uv_run(loop_, UV_RUN_DEFAULT);
        return true;
    }

    // Stop the server and close all connections
    void stop_with_uv() {
        {
            std::lock_guard<std::mutex> lock(conn_mutex_);
            for (auto& conn : connections_) conn->close();
            connections_.clear();
            metrics_.active_connections = 0;
        }
        uv_stop(loop_);
    }

    const ServerMetrics& get_metrics() const { return metrics_; }

private:
    static void pthreadpool_task(void* context, size_t, size_t) {
        auto* task = reinterpret_cast<std::function<void()>*>(context);
        (*task)();
        delete task;
    }

    // Process queued connections in batches
    void process_once() {
        std::vector<StreamPtr> batch;
        batch.reserve(batch_size_);

        while (!conn_queue_.empty()) {
            batch.push_back(conn_queue_.front());
            conn_queue_.erase(conn_queue_.begin());

            if (batch.size() >= batch_size_) {
                dispatch_batch(batch);
                batch.clear();
            }
        }

        if (!batch.empty()) {
            dispatch_batch(batch);
        }
    }

    void dispatch_batch(const std::vector<StreamPtr>& batch) {
        for (auto& s : batch) {
            auto* task = new std::function<void()>([this, s]() {
                try {
                    bool close_conn = false;
                    this->process_request(s, close_conn);

                    // After HTTP handshake, set up async streaming handler
                    s->set_message_handler([this, s](const std::string& msg) {
                        if (s->is_closed()) return;
                        try {
                            LOG_INFO("Server received: " << msg);
                            std::string reply = "Server echo: " + msg;
                            s->write(reply.data(), reply.size());
                        } catch (const std::exception& e) {
                            // metrics_.errors++;
                            LOG_ERROR("Message handler exception: " << e.what());
                        }
                    });

                    if (close_conn) {
                        remove_connection(s);
                    }
                } catch (const std::exception& e) {
                    // metrics_.errors++;
                    LOG_ERROR("process_request exception: " << e.what());
                    remove_connection(s);
                }
            });

            pthreadpool_parallelize_1d(pool_, pthreadpool_task, task, 1, 1);
        }
    }

    void remove_connection(const StreamPtr& stream) {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        if (!stream->is_closed()) {
            stream->close();
            metrics_.active_connections--;
        }
        connections_.erase(stream);
    }

    // ==== Members ====
    std::vector<StreamPtr> conn_queue_;
    uv_loop_t* loop_;
    uv_async_t async_handle_;
    pthreadpool_t pool_;
    size_t max_queue_size_;
    size_t batch_size_;
    std::unordered_set<StreamPtr> connections_;
    std::mutex conn_mutex_;
    ServerMetrics metrics_;
};

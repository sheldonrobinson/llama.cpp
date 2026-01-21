#pragma once

#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <boost/asio.hpp>
#include "httplib.h" // cpp-httplib header
#include "concurrentqueue.h" // moodycamel concurrentqueue

// Forward declaration
class AsyncMemoryStream;

// Link two streams so writes from one go to the other's incoming queue
struct StreamLink {
    std::weak_ptr<AsyncMemoryStream> peer;
};

// In-memory async stream for cpp-httplib
class AsyncMemoryStream : public httplib::Stream, public std::enable_shared_from_this<AsyncMemoryStream> {
public:
    explicit AsyncMemoryStream(boost::asio::io_context& io)
        : io_context_(io), closed_(false) {}

    ~AsyncMemoryStream() override {
        close();
    }

    // Link this stream to another
    void link_peer(const std::shared_ptr<AsyncMemoryStream>& other) {
        peer_link_.peer = other;
    }

    // cpp-httplib read override
    ssize_t read(char* ptr, size_t size) override {
        std::string data;
        if (!incoming_queue_.try_dequeue(data)) {
            return 0; // No data available yet
        }
        size_t to_copy = std::min(size, data.size());
        memcpy(ptr, data.data(), to_copy);
        return static_cast<ssize_t>(to_copy);
    }

    // cpp-httplib write override
    ssize_t write(const char* ptr, size_t size) override {
        if (closed_) return -1;
        auto buf = std::string(ptr, size);

        // Deliver to peer's incoming queue asynchronously
        if (auto peer = peer_link_.peer.lock()) {
            boost::asio::post(io_context_, [peer, buf]() {
                peer->incoming_queue_.enqueue(buf);
            });
        }
        return static_cast<ssize_t>(size);
    }

    void get_remote_ip_and_port(std::string &ip, int &port) const override {
        ip = "127.0.0.1"; // Dummy IP
        port = 0;         // Dummy port
    }

    void close() override {
        closed_ = true;
    }

private:
    boost::asio::io_context& io_context_;
    moodycamel::ConcurrentQueue<std::string> incoming_queue_;
    StreamLink peer_link_;
    std::atomic<bool> closed_;
};

// Utility: Create a linked pair of streams
std::pair<std::shared_ptr<AsyncMemoryStream>, std::shared_ptr<AsyncMemoryStream>>
make_linked_streams(boost::asio::io_context& io) {
    auto a = std::make_shared<AsyncMemoryStream>(io);
    auto b = std::make_shared<AsyncMemoryStream>(io);
    a->link_peer(b);
    b->link_peer(a);
    return {a, b};
}
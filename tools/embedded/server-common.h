#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "mtmd.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <cinttypes>
#include<iostream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)  // Windows
#    include <intrin.h>
#    include <windows.h>

#elif defined(__linux__) && !defined(__ANDROID__)  // Linux
#    include <sys/sysinfo.h>
#    include <unistd.h>

#    include <fstream>
#    include <sstream>

#elif defined(__APPLE__)  // macOS
#    include <mach/mach.h>
#    include <sys/sysctl.h>
#    include <sys/types.h>
#    include <unistd.h>

#elif defined(__ANDROID__)  // Android
#    include <unistd.h>

#    include <fstream>
#    include <sstream>

#else
#    error "Unsupported platform"
#endif

using json = nlohmann::ordered_json;

#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

struct MemoryInfo {
    unsigned long long total_physical;
    unsigned long long available_physical;
    unsigned long long total_swap;
    unsigned long long available_swap;
};

struct CPUInfo {
    std::string architecture;
    int         cores;
    double      frequency_mhz;  // Approximate
};

// ---------------- MEMORY INFO ----------------
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
    info.total_swap         = statex.ullTotalPageFile;
    info.available_swap     = statex.ullAvailPageFile;

#elif defined(__linux__) && !defined(__ANDROID__)
    struct sysinfo memInfo;
    if (sysinfo(&memInfo) != 0) {
        throw std::runtime_error("Failed to get memory info on Linux");
    }
    info.total_physical     = static_cast<unsigned long long>(memInfo.totalram) * memInfo.mem_unit;
    info.available_physical = static_cast<unsigned long long>(memInfo.freeram) * memInfo.mem_unit;
    info.total_swap         = static_cast<unsigned long long>(memInfo.totalswap) * memInfo.mem_unit;
    info.available_swap     = static_cast<unsigned long long>(memInfo.freeswap) * memInfo.mem_unit;

#elif defined(__APPLE__)
    // Total RAM
    int      mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t total;
    size_t   len = sizeof(total);
    if (sysctl(mib, 2, &total, &len, nullptr, 0) != 0) {
        throw std::runtime_error("Failed to get total memory on macOS");
    }
    info.total_physical = total;

    // Available RAM
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    vm_statistics64_data_t vmstat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO, reinterpret_cast<host_info64_t>(&vmstat), &count) !=
        KERN_SUCCESS) {
        throw std::runtime_error("Failed to get available memory on macOS");
    }
    uint64_t free_mem       = static_cast<uint64_t>(vmstat.free_count) * sysconf(_SC_PAGESIZE);
    uint64_t inactive_mem   = static_cast<uint64_t>(vmstat.inactive_count) * sysconf(_SC_PAGESIZE);
    info.available_physical = free_mem + inactive_mem;

    // Swap info
    struct xsw_usage swap;
    size_t           swap_size   = sizeof(swap);
    int              swap_mib[2] = { CTL_VM, VM_SWAPUSAGE };
    if (sysctl(swap_mib, 2, &swap, &swap_size, nullptr, 0) != 0) {
        throw std::runtime_error("Failed to get swap info on macOS");
    }
    info.total_swap     = swap.xsu_total;
    info.available_swap = swap.xsu_avail;

#elif defined(__ANDROID__)
    std::ifstream meminfo_file("/proc/meminfo");
    if (!meminfo_file.is_open()) {
        throw std::runtime_error("Failed to open /proc/meminfo on Android");
    }

    std::string        line;
    unsigned long long mem_total_kb = 0, mem_available_kb = 0;
    unsigned long long swap_total_kb = 0, swap_free_kb = 0;

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
            } else if (key == "SwapTotal:") {
                swap_total_kb = value;
            } else if (key == "SwapFree:") {
                swap_free_kb = value;
            }
        }
    }
    meminfo_file.close();

    info.total_physical     = mem_total_kb * 1024ULL;
    info.available_physical = mem_available_kb * 1024ULL;
    info.total_swap         = swap_total_kb * 1024ULL;
    info.available_swap     = swap_free_kb * 1024ULL;

#endif

    return info;
}

// ---------------- CPU INFO ----------------
CPUInfo get_cpu_info() {
    CPUInfo cpu{};

#if defined(_WIN32)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    cpu.cores = sysInfo.dwNumberOfProcessors;

    // Architecture
    switch (sysInfo.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64:
            cpu.architecture = "x86_64";
            break;
        case PROCESSOR_ARCHITECTURE_INTEL:
            cpu.architecture = "x86";
            break;
        case PROCESSOR_ARCHITECTURE_ARM:
            cpu.architecture = "ARM";
            break;
        case PROCESSOR_ARCHITECTURE_ARM64:
            cpu.architecture = "ARM64";
            break;
        default:
            cpu.architecture = "Unknown";
    }

    // Frequency (MHz)
    DWORD bufSize = sizeof(DWORD);
    DWORD mhz     = 0;
    HKEY  hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) ==
        ERROR_SUCCESS) {
        RegQueryValueExA(hKey, "~MHz", nullptr, nullptr, reinterpret_cast<LPBYTE>(&mhz), &bufSize);
        RegCloseKey(hKey);
    }
    cpu.frequency_mhz = static_cast<double>(mhz);

#elif defined(__linux__) || defined(__ANDROID__)
    cpu.cores = sysconf(_SC_NPROCESSORS_ONLN);

    // Architecture
    struct utsname unameData;
    uname(&unameData);
    cpu.architecture = unameData.machine;

    // Frequency from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string   line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            std::string        label;
            double             mhz;
            char               colon;
            std::istringstream iss(line);
            if (iss >> label >> colon >> mhz) {
                cpu.frequency_mhz = mhz;
                break;  // Take the first CPU's frequency
            }
        }
    }
    cpuinfo.close();

#elif defined(__APPLE__)
    // Core count
    int    cores;
    size_t size = sizeof(cores);
    if (sysctlbyname("hw.ncpu", &cores, &size, nullptr, 0) != 0) {
        throw std::runtime_error("Failed to get CPU core count on macOS");
    }
    cpu.cores = cores;

    // Architecture
    char arch[256];
    size = sizeof(arch);
    if (sysctlbyname("hw.machine", &arch, &size, nullptr, 0) != 0) {
        throw std::runtime_error("Failed to get CPU architecture on macOS");
    }
    cpu.architecture = arch;

    // Frequency (Hz → MHz)
    uint64_t freq;
    size = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency", &freq, &size, nullptr, 0) != 0) {
        throw std::runtime_error("Failed to get CPU frequency on macOS");
    }
    cpu.frequency_mhz = static_cast<double>(freq) / 1'000'000.0;

#endif

    return cpu;
}

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const & err) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value: %s\n", key.c_str(), json(default_value).type_name(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

json format_error_response(const std::string & message, const enum error_type type);

//
// random string / id
//

std::string random_string();
std::string gen_chatcmplid();
std::string gen_tool_call_id();

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next);

std::map<int, float> parse_lora_request(const json & data);

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2);

// get the ids of all enabled loras
std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras);

//
// server_tokens
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
    // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
    // idx  0   1   2   3   4   5      6      7      8      9      10
    // pos  0   1   2   3   4   5      5      5      7      7      7
    // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    // TODO: server_tokens should be copyable - remove this:
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd);
    server_tokens(const llama_tokens & tokens, bool has_mtmd);

    // for debugging
    std::string str() const;

    llama_pos pos_next() const;
    const mtmd::input_chunk_ptr & find_chunk(size_t idx) const;

    void push_back(llama_token tok);

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk * chunk);

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens & tokens);

    // for compatibility with context shift and prompt truncation
    void insert(const llama_tokens & inp_tokens);

    // for compatibility with speculative decoding, ctx shift, slot save/load
    const llama_tokens & get_text_tokens() const;

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id);

    size_t size() const { return tokens.size(); }

    bool empty() const { return tokens.empty(); }

    void clear() {
        map_idx_to_media.clear();
        tokens.clear();
    }

    void keep_first(size_t n);

    std::string detokenize(const llama_context * ctx, bool special) const;

    size_t get_common_prefix(const server_tokens & b) const;

    // make sure all text tokens are within the vocab range
    bool validate(const struct llama_context * ctx) const;

    // encode and decode the image chunk
    int32_t process_chunk(
                llama_context * ctx,
                mtmd_context * mctx,
                size_t idx,
                llama_pos pos,
                int32_t seq_id,
                size_t & n_tokens_out) const;

    server_tokens clone() const;
};


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json & data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json & data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string> & paths, const json & js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
server_tokens process_mtmd_prompt(mtmd_context * mctx, std::string prompt, std::vector<raw_buffer> files);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(
                                        const llama_vocab * vocab,
                                        mtmd_context * mctx,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs; // mapping key --> json value
    common_chat_templates_ptr tmpls;
    bool allow_image;
    bool allow_audio;
    bool enable_thinking = true;
    std::string media_path;
};

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body);

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files);

// convert OpenAI Responses API format to OpenAI Chat Completions API format
json convert_responses_to_chatcmpl(const json & body);

// convert Anthropic Messages API format to OpenAI Chat Completions API format
json convert_anthropic_to_oai(const json & body);

// TODO: move it to server-task.cpp
json format_embeddings_response_oaicompat(
    const json & request,
    const std::string & model_name,
    const json & embeddings,
    bool use_base64 = false);

// TODO: move it to server-task.cpp
json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n);

//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx);

std::string safe_json_to_str(const json & data);

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens);
std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens);

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data);

std::string format_oai_resp_sse(const json & data);

// format Anthropic-style SSE with event types
std::string format_anthropic_sse(const json & data);

bool is_valid_utf8(const std::string & str);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc);

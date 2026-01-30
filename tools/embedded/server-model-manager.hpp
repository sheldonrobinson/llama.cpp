#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <shared_mutex>
#include <mutex>
#include <chrono>
#include <deque>
#include <vector>
#include <algorithm>
#include <functional>
#include <nlohmann/json.hpp>
#include "llama.h"
#include "common.h"

#include "server-embedded.h"

inline std::string stateToString(server_model_status s) {
    switch (s) {
        case SERVER_MODEL_STATUS_LOADING:      return "LOADING";
        case SERVER_MODEL_STATUS_LOADED:       return "LOADED";
        case SERVER_MODEL_STATUS_UNLOADING:    return "UNLOADING";
        case SERVER_MODEL_STATUS_UNLOADED:     return "UNLOADED";
        case SERVER_MODEL_STATUS_IDLE:         return "IDLE";
        case SERVER_MODEL_STATUS_PAUSED:       return "PAUSED";
        case SERVER_MODEL_STATUS_RESUMING:     return "RESUMING";
        case SERVER_MODEL_STATUS_ERROR:        return "ERROR";
        case SERVER_MODEL_STATUS_FAILED:       return "FAILED";
    }
    return "UNKNOWN";
}

struct ModelContext {
    std::shared_ptr<llama_model> model;
    std::shared_ptr<llama_context> context;
    bool active = false;
    server_model_status state = SERVER_MODEL_STATUS_UNLOADED;
    size_t memoryUsageMB = 0;
    int fitted_n_ctx = 0;
    int fitted_n_batch = 0;
    int fitted_n_gpu_layers = 0;
    std::chrono::system_clock::time_point lastStateChange;
    std::deque<std::string> history;
    std::mutex inferenceMutex;
};

class ModelManager {
public:
    using StateChangeListener = std::function<void(const std::string &, server_model_status, server_model_status)>;

    explicit ModelManager(size_t maxMemMB, size_t baseTenantQuotaMB = 4096)
        : maxMemoryMB(maxMemMB), baseTenantQuotaMB(baseTenantQuotaMB), currentMemoryMB(0) {}

    void addStateChangeListener(StateChangeListener listener) {
        std::lock_guard<std::mutex> lock(listenerMutex);
        listeners.push_back(std::move(listener));
    }

    void loadModel(const std::string &tenantModelName, const common_params &params) {
        std::unique_lock<std::shared_mutex> lock(globalMutex);

        if (!std::filesystem::exists(params.model)) {
            throw std::runtime_error("Model file not found: " + params.model);
        }
        if (models.count(tenantModelName)) {
            throw std::runtime_error("Model already loaded: " + tenantModelName);
        }

        llama_model_params mparams = llama_model_default_params();
        mparams.n_ctx        = params.n_ctx;
        mparams.n_gpu_layers = params.n_gpu_layers;
        mparams.use_mmap     = params.use_mmap;
        mparams.use_mlock    = params.use_mlock;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = params.n_ctx;
        cparams.n_threads = params.n_threads;
        cparams.n_batch   = params.n_batch;

        llama_params_fit(&mparams, &cparams);

        size_t estimatedMem = estimateModelMemory(params.model);
        std::string tenant = tenantFromKey(tenantModelName);

        if (currentMemoryMB + estimatedMem > maxMemoryMB) {
            evictLRU(estimatedMem, tenant);
        }

        size_t effectiveQuota = getEffectiveTenantQuota(tenant);
        if (tenantMemoryUsage[tenant] + estimatedMem > effectiveQuota) {
            throw std::runtime_error("Tenant quota exceeded for " + tenant +
                                     " (quota=" + std::to_string(effectiveQuota) + "MB)");
        }

        ModelContext ctx;
        changeState(tenantModelName, ctx, SERVER_MODEL_STATUS_LOADING);
        models[tenantModelName] = std::move(ctx);

        llama_model *rawModel = llama_load_model_from_file(params.model.c_str(), mparams);
        if (!rawModel) {
            changeState(tenantModelName, models[tenantModelName], SERVER_MODEL_STATUS_FAILED);
            throw std::runtime_error("Failed to load model: " + params.model);
        }
        std::shared_ptr<llama_model> modelPtr(rawModel, [](llama_model *m) { llama_free_model(m); });

        llama_context *rawCtx = llama_new_context_with_model(modelPtr.get(), cparams);
        if (!rawCtx) {
            changeState(tenantModelName, models[tenantModelName], SERVER_MODEL_STATUS_FAILED);
            throw std::runtime_error("Failed to create context for model: " + tenantModelName);
        }
        std::shared_ptr<llama_context> ctxPtr(rawCtx, [](llama_context *ctx) { llama_free(ctx); });

        models[tenantModelName].model = modelPtr;
        models[tenantModelName].context = ctxPtr;
        models[tenantModelName].active = false;
        models[tenantModelName].memoryUsageMB = estimatedMem;
        models[tenantModelName].fitted_n_ctx = cparams.n_ctx;
        models[tenantModelName].fitted_n_batch = cparams.n_batch;
        models[tenantModelName].fitted_n_gpu_layers = mparams.n_gpu_layers;

        changeState(tenantModelName, models[tenantModelName], SERVER_MODEL_STATUS_LOADED);

        currentMemoryMB += estimatedMem;
        tenantMemoryUsage[tenant] += estimatedMem;
        tenantQuotaBoost[tenant] = 0;
    }

    void unloadModel(const std::string &tenantModelName) {
        std::unique_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) throw std::runtime_error("Model not found");

        changeState(tenantModelName, it->second, SERVER_MODEL_STATUS_UNLOADING);

        currentMemoryMB -= it->second.memoryUsageMB;
        tenantMemoryUsage[tenantFromKey(tenantModelName)] -= it->second.memoryUsageMB;
        models.erase(it);
    }

    void setActiveModel(const std::string &tenantModelName, bool activeFlag) {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) throw std::runtime_error("Model not found");
        changeState(tenantModelName, it->second, activeFlag ? SERVER_MODEL_STATUS_IDLE : SERVER_MODEL_STATUS_LOADED);
        it->second.active = activeFlag;
    }

    void pauseModel(const std::string &tenantModelName) {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) throw std::runtime_error("Model not found");
        changeState(tenantModelName, it->second, SERVER_MODEL_STATUS_PAUSED);
    }

    void resumeModel(const std::string &tenantModelName) {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) throw std::runtime_error("Model not found");
        changeState(tenantModelName, it->second, SERVER_MODEL_STATUS_RESUMING);
    }

    void setErrorState(const std::string &tenantModelName) {
        std::shared_lock<std::shared_mutex>
        auto it = models.find(tenantModelName);
        if (it != models.end()) {
            changeState(tenantModelName, it->second, SERVER_MODEL_STATUS_ERROR);
        }
    }

    server_model_status getModelState(const std::string &tenantModelName) const {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) throw std::runtime_error("Model not found");
        return it->second.state;
    }

    ModelContext &getModelContext(const std::string &tenantModelName) {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) throw std::runtime_error("Model not found");
        return it->second;
    }

    std::vector<std::string> listModels() const {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
		std::vector<std::string> _models;
        for (auto &p : models) {
			std::string model(p.first);
			model.append((p.second.active ? "-ACTIVE;" : "-INACTIVE;"))
				.append(stateToString(p.second.state)).append(";")
				.append(p.second.memoryUsageMB).append("MB")
            _models.add(model);
        }
		return _models;
    }

    nlohmann::json listModelsJson() const {
        nlohmann::json j = nlohmann::json::array();
        std::shared_lock<std::shared_mutex> lock(globalMutex);

        for (auto &p : models) {
            std::string tenant = tenantFromKey(p.first);

            nlohmann::json m;
            m["name"] = p.first;
            m["tenant"] = tenant;
            m["active"] = p.second.active;
            m["state"] = stateToString(p.second.state);
            m["memory_mb"] = p.second.memoryUsageMB;
            m["last_change_sec_ago"] =
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - p.second.lastStateChange
                ).count();
            m["history"] = p.second.history;
            m["fitted_params"] = {
                {"n_ctx", p.second.fitted_n_ctx},
                {"n_batch", p.second.fitted_n_batch},
                {"n_gpu_layers", p.second.fitted_n_gpu_layers}
            };

            size_t usage = tenantMemoryUsage.count(tenant) ? tenantMemoryUsage.at(tenant) : 0;
            size_t boost = tenantQuotaBoost.count(tenant) ? tenantQuotaBoost.at(tenant) : 0;

            m["tenant_quota"] = {
                {"effective_quota_mb", getEffectiveTenantQuota(tenant)},
                {"base_quota_mb", baseTenantQuotaMB},
                {"quota_boost_mb", boost},
                {"current_usage_mb", usage}
            };

            j.push_back(m);
        }
        return j;
    }

private:
    std::unordered_map<std::string, ModelContext> models;
    mutable std::shared_mutex globalMutex;
    size_t maxMemoryMB;
    size_t baseTenantQuotaMB;
    size_t currentMemoryMB;

    std::unordered_map<std::string, size_t> tenantMemoryUsage;
    std::unordered_map<std::string, size_t> tenantQuotaBoost;
    std::vector<StateChangeListener> listeners;
    mutable std::mutex listenerMutex;

    inline std::string tenantFromKey(const std::string &key) const {
        auto pos = key.find(':');
        return (pos != std::string::npos) ? key.substr(0, pos) : "default";
    }

    size_t getEffectiveTenantQuota(const std::string &tenant) const {
        size_t activeTenants = tenantMemoryUsage.size();
        size_t globalFreeMB = (maxMemoryMB > currentMemoryMB) ? (maxMemoryMB - currentMemoryMB) : 0;
        size_t baseQuota = baseTenantQuotaMB + (activeTenants > 0 ? globalFreeMB / activeTenants : globalFreeMB);
        size_t boost = tenantQuotaBoost.count(tenant) ? tenantQuotaBoost.at(tenant) : 0;
        return baseQuota + boost;
    }

    void changeState(const std::string &name, ModelContext &ctx, server_model_status newState) {
        server_model_status oldState = ctx.state;
        ctx.state = newState;
        ctx.lastStateChange = std::chrono::system_clock::now();
        ctx.history.push_back(stateToString(newState));

        std::vector<StateChangeListener> copy;
        {
            std::lock_guard<std::mutex> lock(listenerMutex);
            copy = listeners;
        }
        for (auto &cb : copy) {
            try {
                cb(name, oldState, newState);
            } catch (...) {}
        }
    }

    size_t estimateModelMemory(const std::string &path) const {
        try {
            return static_cast<size_t>(std::filesystem::file_size(path) / (1024 * 1024));
        } catch (...) {
            return 512;
        }
    }

    void evictLRU(size_t neededMB, const std::string &requestingTenant) {
        std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> sameTenant;
        std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> otherTenant;

        for (auto &kv : models) {
            if (!kv.second.active) {
                std::string tenant = tenantFromKey(kv.first);
                if (tenant == requestingTenant) {
                    sameTenant.push_back({kv.first, kv.second.lastStateChange});
                } else {
                    otherTenant.push_back({kv.first, kv.second.lastStateChange});
                }
            }
        }

        auto sortByLRU = [](auto &a, auto &b) { return a.second < b.second; };
        std::sort(sameTenant.begin(), sameTenant.end(), sortByLRU);
        std::sort(otherTenant.begin(), otherTenant.end(), sortByLRU);

        bool freedEnough = false;

        for (auto &c : sameTenant) {
            size_t memBefore = currentMemoryMB;
            unloadModel(c.first);
            size_t freed = memBefore - currentMemoryMB;
            tenantQuotaBoost[requestingTenant] += freed;
            if (currentMemoryMB + neededMB <= maxMemoryMB) {
                freedEnough = true;
                break;
            }
        }

        if (!freedEnough) {
            for (auto &c : otherTenant) {
                std::string tenant = tenantFromKey(c.first);
                size_t memBefore = currentMemoryMB;
                unloadModel(c.first);
                size_t freed = memBefore - currentMemoryMB;
                tenantQuotaBoost[tenant] += freed;
                if (currentMemoryMB + neededMB <= maxMemoryMB) {
                    freedEnough = true;
                    break;
                }
            }
        }

        if (!freedEnough) {
            throw std::runtime_error(
                "Not enough memory to load model, even after tenant-aware eviction"
            );
        }
    }

};

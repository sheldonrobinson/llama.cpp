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
#include "server-context.h"

#include "server-embedded.h"

struct ModelContext {
    // Default constructor
    ModelContext() = default;

    // User-defined copy constructor
    ModelContext(const ModelContext & other) :
        server_ctx(other.server_ctx),
        state(other.state),
        memoryUsageMB(other.memoryUsageMB),
        fitted_n_ctx(other.fitted_n_ctx),
        fitted_n_batch(other.fitted_n_batch),
        fitted_n_gpu_layers(other.fitted_n_gpu_layers),
        lastStateChange(other.lastStateChange),
        history(other.history) {}

    std::shared_ptr<server_context> server_ctx = std::make_shared<server_context>();
    server_model_status_t state = server_model_status::SERVER_MODEL_STATUS_UNLOADED;
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

    void setMaxMemory(size_t maxMemMB) { maxMemoryMB = maxMemMB;}

    void addStateChangeListener(StateChangeListener listener) {
        std::lock_guard<std::mutex> lock(listenerMutex);
        listeners.push_back(std::move(listener));
    }
	
	void clearAllStateChangeListeners() {
        std::lock_guard<std::mutex> lock(listenerMutex);
        listeners.clear();
    }

    void loadModel(const std::string &tenantModelName, const common_params &params) {
        std::unique_lock<std::shared_mutex> lock(globalMutex);

        if (!std::filesystem::exists(params.model.path)) {
            throw std::runtime_error("Model file not found: " + params.model.path);
        }
        if (models.count(tenantModelName)) {
            SRV_WRN("Model already loaded: %s", tenantModelName.c_str());
            return;
        }

		common_params args = params;
		llama_model_params   mparams = common_model_params_to_llama(args);
		llama_context_params cparams = common_context_params_to_llama(args);

		const llama_params_fit_status status = llama_params_fit(
			args.model.path.c_str(), &mparams, &cparams, args.tensor_split, args.tensor_buft_overrides.data(), args.fit_params_target.data(), args.fit_params_min_ctx,
							 args.verbosity >= 4 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
		if (status != LLAMA_PARAMS_FIT_STATUS_SUCCESS) {
			SRV_WRN("Failed to determine llama_params_fit, using defaults for model %s", params.model.path.c_str());
		}

        args.n_ctx = cparams.n_ctx;
        args.n_batch = cparams.n_batch;
        args.n_gpu_layers = mparams.n_gpu_layers;
        args.fit_params   = false;  // fitted already
        args.split_mode   = mparams.split_mode;
        if (mparams.tensor_buft_overrides) {
            args.tensor_buft_overrides.clear();
            for (int i = 0; (mparams.tensor_buft_overrides + i) != NULL; i++) {
                args.tensor_buft_overrides.push_back(*(mparams.tensor_buft_overrides + i));
            }
        }
 
		
        if (mparams.split_mode != llama_split_mode::LLAMA_SPLIT_MODE_NONE) {
            size_t          nd           = llama_max_devices();
            float          allocatedSum = 0.0f;
			float		   lp2normSum = 0.0f;
            constexpr float LAST_EPSILON = 1.0f - FLT_EPSILON;
			size_t count = sizeof(mparams.tensor_split) / sizeof(mparams.tensor_split[0]);
			size_t N = count < nd ? count : nd;
            for (int i = 0; i < N; i++) {
				float val = fabs(*(mparams.tensor_split + i));
				args.tensor_split[i] = val;
				allocatedSum += val;
				lp2normSum += (val * val);
            }
            // rescale
            if (lp2normSum > 0.0f && fabs(1.0f - allocatedSum) > FLT_EPSILON) {
                for (size_t j = 0; j < N; j++) {
                    args.tensor_split[j] = (args.tensor_split[j] * args.tensor_split[j])/lp2normSum;
                }
            }
        }
       

        size_t estimatedMem = estimateModelMemory(params.model.path);
        std::string tenant = tenantFromKey(tenantModelName);

        if (currentMemoryMB + estimatedMem > maxMemoryMB) {
            evictLRU(estimatedMem, tenant);
        }

        size_t effectiveQuota = getEffectiveTenantQuota(tenant);
        if (tenantMemoryUsage[tenant] + estimatedMem > effectiveQuota) {
            throw std::runtime_error("Tenant quota exceeded for " + tenant +
                                     " (quota=" + std::to_string(effectiveQuota) + "MB)");
        }
        auto & result = models.emplace(std::make_pair(tenantModelName, ModelContext()));

        changeState(tenantModelName, result.first->second, server_model_status::SERVER_MODEL_STATUS_LOADING);
		
		ModelContext ctx = models[tenantModelName];
		bool is_model_loaded = ctx.server_ctx->load_model(args);
        if (!is_model_loaded) {
            throw std::runtime_error("Failed to load model: " + params.model.path);
        }

        models[tenantModelName].memoryUsageMB = estimatedMem;
        models[tenantModelName].fitted_n_ctx = cparams.n_ctx;
        models[tenantModelName].fitted_n_batch = cparams.n_batch;
        models[tenantModelName].fitted_n_gpu_layers = mparams.n_gpu_layers;

        changeState(tenantModelName, models[tenantModelName], server_model_status::SERVER_MODEL_STATUS_LOADED);

        currentMemoryMB += estimatedMem;
        tenantMemoryUsage[tenant] += estimatedMem;
        tenantQuotaBoost[tenant] = 0;
    }

    void unloadModel(const std::string &tenantModelName) {
        std::unique_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) return;
        ModelContext& model_ctx_ref = it->second;
        auto server_ctx = model_ctx_ref.server_ctx;
        server_ctx->terminate();
        changeState(tenantModelName, model_ctx_ref, server_model_status::SERVER_MODEL_STATUS_UNLOADED);

        currentMemoryMB -= model_ctx_ref.memoryUsageMB;
        tenantMemoryUsage[tenantFromKey(tenantModelName)] -= model_ctx_ref.memoryUsageMB;
        models.erase(it);
    }

    server_model_status_t getModelState(const std::string &tenantModelName) const {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()) return server_model_status::SERVER_MODEL_STATUS_UNLOADED;
        return it->second.state;
    }

    ModelContext& getModelContext(const std::string &tenantModelName) {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
        auto it = models.find(tenantModelName);
        if (it == models.end()){
			ModelContext ctx;
			return ctx;
		}
        return it->second;
    }

    std::vector<std::string> listModels() const {
        std::shared_lock<std::shared_mutex> lock(globalMutex);
		std::vector<std::string> _models;
        for (auto &p : models) {
			std::string model(p.first);
			model.append(";")
				.append(server_model_status_to_string(p.second.state))
				.append(";")
				.append(std::to_string(p.second.memoryUsageMB))
				.append("MB");
            _models.push_back(model);
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
            m["state"] = server_model_status_to_string(p.second.state);
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
        ctx.history.push_back(server_model_status_to_string(newState));

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
			std::string tenant = tenantFromKey(kv.first);
			if (tenant == requestingTenant) {
				sameTenant.push_back({kv.first, kv.second.lastStateChange});
			} else {
				otherTenant.push_back({kv.first, kv.second.lastStateChange});
			}
        }

        auto sortByLRU = [](auto &a, auto &b) { return a.second < b.second; };
        std::sort(sameTenant.begin(), sameTenant.end(), sortByLRU);
        std::sort(otherTenant.begin(), otherTenant.end(), sortByLRU);

        bool freedEnough = false;

        for (auto &kv : models) {
			std::string tenant = tenantFromKey(kv.first);
			if (tenant == requestingTenant) {
				sameTenant.push_back({kv.first, kv.second.lastStateChange});
			} else {
				otherTenant.push_back({kv.first, kv.second.lastStateChange});
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

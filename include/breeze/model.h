#pragma once

#include "breeze/config.h"
#include "breeze/common.h"
#include "breeze/gguf_loader.h"
#include "breeze/tokenizer.h"

#include <memory>
#include <string>

namespace breeze {

struct BreezeModel {
    Backend backend;
    GGUFModel gg;
    BreezeConfig cfg;
    Tokenizer tok;

    const BreezeModel * base_model = nullptr;
    const GGUFModel * dd_override = nullptr;

    bool load(const std::string & path, bool prefer_gpu);
    bool load_device(const std::string & path, int cuda_device);
    void free();

    ggml_tensor * w(const std::string & name) const { 
        if (dd_override && dd_override->has(name)) {
            return dd_override->get(name);
        }
        if (base_model) {
            return base_model->w(name);
        }
        return gg.get(name); 
    }
    ggml_tensor * wopt(const std::string & name) const { 
        if (dd_override && dd_override->has(name)) {
            return dd_override->find(name);
        }
        if (base_model) {
            return base_model->wopt(name);
        }
        return gg.find(name); 
    }
};

}

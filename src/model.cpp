#include "breeze/model.h"

namespace breeze {

bool BreezeModel::load_device(const std::string & path, int cuda_device) {
    backend.init_device(cuda_device);
    if (!gg.load(path, backend)) return false;
    cfg = parse_config(gg);
    if (!tok.load(gg)) return false;
    return true;
}

bool BreezeModel::load(const std::string & path, bool prefer_gpu) {
    backend.init(prefer_gpu);
    if (!gg.load(path, backend)) return false;
    cfg = parse_config(gg);
    if (!tok.load(gg)) return false;
    return true;
}

void BreezeModel::free() {
    gg.free();
    backend.free();
}

}

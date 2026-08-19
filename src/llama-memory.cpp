#include "llama-memory.h"
#include <cstdlib>
#include <string>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

// parse one "TYPEK[/TYPEV][:wN][:rot|:norot]" fragment into cfg
static bool llama_kv_layer_cfg_parse_one(const std::string & s, llama_kv_layer_cfg & cfg, std::string & err) {
    // split off the ':'-separated suffix options; the leading part is the type spec
    size_t colon = s.find(':');
    std::string rest = s.substr(0, colon == std::string::npos ? s.size() : colon);

    while (colon != std::string::npos) {
        const size_t next = s.find(':', colon + 1);
        const std::string opt = s.substr(colon + 1, (next == std::string::npos ? s.size() : next) - colon - 1);
        if (opt.size() > 1 && opt[0] == 'w') {
            char * end = nullptr;
            const unsigned long w = strtoul(opt.c_str() + 1, &end, 10);
            if (end && *end != '\0') {
                err = "bad window '" + opt + "'";
                return false;
            }
            cfg.window = (uint32_t) w;
        } else if (opt == "rot") {
            cfg.rot = 1;
        } else if (opt == "norot") {
            cfg.rot = 0;
        } else {
            err = "unknown option ':" + opt + "'";
            return false;
        }
        colon = next;
    }

    auto parse_type = [&](const std::string & name, ggml_type & t) {
        if (name.empty()) {
            return true; // inherit
        }
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            if (name == ggml_type_name((ggml_type) i)) {
                t = (ggml_type) i;
                return true;
            }
        }
        err = "unknown ggml type '" + name + "'";
        return false;
    };

    const size_t slash = rest.find('/');
    if (slash == std::string::npos) {
        // one type = both K and V
        if (!parse_type(rest, cfg.type_k)) {
            return false;
        }
        cfg.type_v = cfg.type_k;
        return true;
    }

    return parse_type(rest.substr(0, slash), cfg.type_k) &&
           parse_type(rest.substr(slash + 1), cfg.type_v);
}

bool llama_kv_layer_cfg_parse(const char * spec, uint32_t n_layer, llama_kv_layer_cfg_map & out, std::string & err) {
    std::string s(spec);

    size_t beg = 0;
    while (beg < s.size()) {
        size_t end = s.find(',', beg);
        if (end == std::string::npos) {
            end = s.size();
        }
        const std::string entry = s.substr(beg, end - beg);
        beg = end + 1;

        if (entry.empty()) {
            continue;
        }

        const size_t eq = entry.find('=');
        if (eq == std::string::npos) {
            err = "entry '" + entry + "' has no '='";
            return false;
        }

        const std::string il_str = entry.substr(0, eq);

        uint32_t il;
        if (il_str == "*") {
            il = LLAMA_KV_LAYER_ALL;
        } else {
            char * endp = nullptr;
            const unsigned long v = strtoul(il_str.c_str(), &endp, 10);
            if (il_str.empty() || (endp && *endp != '\0') || v >= n_layer) {
                err = "bad layer index '" + il_str + "' (n_layer = " + std::to_string(n_layer) + ")";
                return false;
            }
            il = (uint32_t) v;
        }

        llama_kv_layer_cfg cfg;
        if (!llama_kv_layer_cfg_parse_one(entry.substr(eq + 1), cfg, err)) {
            return false;
        }

        out[il] = cfg;
    }

    return true;
}

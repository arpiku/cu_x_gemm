#pragma once

enum class TargetArch {
    H100,
    RTX5070,
};

inline const char* target_arch_name(TargetArch arch) {
    switch (arch) {
        case TargetArch::H100: return "h100";
        case TargetArch::RTX5070: return "rtx5070";
    }
    return "unknown";
}

inline bool parse_target_arch_flag(const char* arg, TargetArch* out) {
    if (!arg || !out) return false;
    if (arg[0] != '-') return false;
    if (arg[1] == 'h' && arg[2] == '1' && arg[3] == '0' && arg[4] == '0' && arg[5] == '\0') {
        *out = TargetArch::H100;
        return true;
    }
    if (arg[1] == 'r' && arg[2] == 't' && arg[3] == 'x' && arg[4] == '5' && arg[5] == '0' &&
        arg[6] == '7' && arg[7] == '0' && arg[8] == '\0') {
        *out = TargetArch::RTX5070;
        return true;
    }
    return false;
}

#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <cuda.h>
#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "settings.h"

#define READ 0
#define WRITE 1
#define MIXED 2

struct Settings {
    uint32_t cudaDevice;
    uint64_t cudaDeviceId;
    const char* blockDevicePath;
    const char* controllerPath;
    uint64_t controllerId;
    uint32_t adapter;
    uint32_t segmentId;
    uint32_t nvmNamespace;
    bool doubleBuffered;
    size_t numReqs;
    size_t numPages;
    size_t startBlock;
    bool stats;
    const char* output;
    size_t numThreads;
    uint32_t domain;
    uint32_t bus;
    uint32_t devfn;
    size_t blkSize;
    size_t queueDepth;
    size_t numQueues;
    size_t pageSize;
    uint64_t numBlks;
    bool random;
    uint64_t accessType;
    uint64_t ratio;
    uint64_t ssdtype;
    // 图节点重排
    bool graph_reorganize;
    const char* graph_reorg_node_map_path;
    // 邻居节点放一起
    bool neighbor_feature;
    const char* neighbor_feature_node_list_path;
    const char* neighbor_len_path;

    const char* input;
    const char* node_sequence;
    const char* batch_size;
    uint64_t num_page;
    uint64_t cache_size;
    const char* libnvmName;
    Settings();
    void parseArguments(int argc, char** argv);

    static std::string usageString(const std::string& name);

    std::string getDeviceBDF() const;
};

struct OptionIface;
using std::make_shared;
using std::string;
using std::vector;
typedef std::shared_ptr<OptionIface> OptionPtr;
typedef std::map<int, OptionPtr> OptionMap;

struct OptionIface {
    const char* type;
    const char* name;
    const char* description;
    const char* defaultValue;
    int hasArgument;

    virtual ~OptionIface() = default;

    OptionIface(const char* type, const char* name, const char* description) : type(type), name(name), description(description), hasArgument(no_argument) {}

    OptionIface(const char* type, const char* name, const char* description, const char* dvalue)
        : type(type), name(name), description(description), defaultValue(dvalue), hasArgument(no_argument) {}

    virtual void parseArgument(const char* optstr, const char* optarg) = 0;

    virtual void throwError(const char*, const char* optarg) const {
        throw string("Option ") + name + string(" expects a ") + type + string(", but got `") + optarg + string("'");
    }
};

template <typename T>
struct Option : public OptionIface {
    T& value;

    Option() = delete;
    Option(Option&& rhs) = delete;
    Option(const Option& rhs) = delete;

    Option(T& value, const char* type, const char* name, const char* description) : OptionIface(type, name, description), value(value) {
        hasArgument = required_argument;
    }

    Option(T& value, const char* type, const char* name, const char* description, const char* dvalue)
        : OptionIface(type, name, description, dvalue), value(value) {
        hasArgument = required_argument;
    }

    void parseArgument(const char* optstr, const char* optarg) override;
};

template <>
void Option<uint32_t>::parseArgument(const char* optstr, const char* optarg) {
    char* endptr = nullptr;

    value = std::strtoul(optarg, &endptr, 0);

    if (endptr == nullptr || *endptr != '\0') {
        throwError(optstr, optarg);
    }
}

template <>
void Option<uint64_t>::parseArgument(const char* optstr, const char* optarg) {
    char* endptr = nullptr;

    value = std::strtoul(optarg, &endptr, 0);

    if (endptr == nullptr || *endptr != '\0') {
        throwError(optstr, optarg);
    }
}

template <>
void Option<bool>::parseArgument(const char* optstr, const char* optarg) {
    string str(optarg);
    std::transform(str.begin(), str.end(), str.begin(), std::ptr_fun<int, int>(std::tolower));

    if (str == "false" || str == "0" || str == "no" || str == "n" || str == "off" || str == "disable" || str == "disabled") {
        value = false;
    } else if (str == "true" || str == "1" || str == "yes" || str == "y" || str == "on" || str == "enable" || str == "enabled") {
        value = true;
    } else {
        throwError(optstr, optarg);
    }
}

template <>
void Option<const char*>::parseArgument(const char*, const char* optarg) {
    value = optarg;
}

struct Range : public Option<uint64_t> {
    uint64_t lower;
    uint64_t upper;

    Range(uint64_t& value, uint64_t lo, uint64_t hi, const char* name, const char* description, const char* dv)
        : Option<uint64_t>(value, "count", name, description, dv), lower(lo), upper(hi) {}

    void throwError(const char*, const char*) const override {
        if (upper != 0 && lower != 0) {
            throw string("Option ") + name + string(" expects a value between ") + std::to_string(lower) + " and " + std::to_string(upper);
        } else if (lower != 0) {
            throw string("Option ") + name + string(" must be at least ") + std::to_string(lower);
        }
        throw string("Option ") + name + string(" must lower than ") + std::to_string(upper);
    }

    void parseArgument(const char* optstr, const char* optarg) override {
        Option<uint64_t>::parseArgument(optstr, optarg);

        if (lower != 0 && value < lower) {
            throwError(optstr, optarg);
        }

        if (upper != 0 && value > upper) {
            throwError(optstr, optarg);
        }
    }
};

static void setBDF(Settings& settings) {
    cudaDeviceProp props;

    cudaError_t err = cudaGetDeviceProperties(&props, settings.cudaDevice);
    if (err != cudaSuccess) {
        throw string("Failed to get device properties: ") + cudaGetErrorString(err);
    }

    settings.domain = props.pciDomainID;
    settings.bus = props.pciBusID;
    settings.devfn = props.pciDeviceID;
}

string Settings::getDeviceBDF() const {
    using namespace std;
    ostringstream s;

    s << setfill('0') << setw(4) << hex << domain << ":" << setfill('0') << setw(2) << hex << bus << ":" << setfill('0') << setw(2) << hex << devfn << ".0";

    return s.str();
}

string Settings::usageString(const string& name) {
    // return "Usage: " + name + " --ctrl=identifier [options]\n"
    //+  "   or: " + name + " --block-device=path [options]";
    return "\n";
}

static string helpString(const string& /*name*/, OptionMap& options) {
    using namespace std;
    ostringstream s;

    s << "" << left << setw(16) << "OPTION" << setw(2) << " " << setw(16) << "TYPE" << setw(10) << "DEFAULT" << setw(36) << "DESCRIPTION" << endl;

    for (const auto& optPair : options) {
        const auto& opt = optPair.second;
        s << "  " << left << setw(16) << opt->name << setw(16) << opt->type << setw(10) << (opt->defaultValue != nullptr ? opt->defaultValue : "") << setw(36)
          << opt->description << endl;
    }

    return s.str();
}

static void createLongOptions(vector<option>& options, string& optionString, const OptionMap& parsers) {
    options.push_back(option{.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'});
    optionString = ":h";

    for (const auto& parserPair : parsers) {
        int shortOpt = parserPair.first;
        const OptionPtr& parser = parserPair.second;

        option opt;
        opt.name = parser->name;
        opt.has_arg = parser->hasArgument;
        opt.flag = nullptr;
        opt.val = shortOpt;

        options.push_back(opt);

        if ('0' <= shortOpt && shortOpt <= 'z') {
            optionString += (char)shortOpt;
            if (parser->hasArgument == required_argument) {
                optionString += ":";
            }
        }
    }

    options.push_back(option{.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0});
}

static void verifyCudaDevice(int device) {
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        throw string("Unexpected error: ") + cudaGetErrorString(err);
    }

    if (device < 0 || device >= deviceCount) {
        throw string("Invalid CUDA device: ") + std::to_string(device);
    }
}

static void verifyNumberOfThreads(size_t numThreads) {
    size_t i = 0;

    while ((1ULL << i) <= 32) {
        if ((1ULL << i) == numThreads) {
            return;
        }

        ++i;
    }

    throw string("Invalid number of threads, must be a power of 2");
}

void Settings::parseArguments(int argc, char** argv) {
    OptionMap parsers = {
#ifdef __DIS_CLUSTER__
        {'c', OptionPtr(new Option<uint64_t>(controllerId, "fdid", "ctrl", "NVM controller device identifier"))},
        {'f', OptionPtr(new Option<uint64_t>(cudaDeviceId, "fdid", "fdid", "CUDA device FDID"))},
        {'a', OptionPtr(new Option<uint32_t>(adapter, "number", "adapter", "DIS adapter number", "0"))},
        {'S', OptionPtr(new Option<uint32_t>(segmentId, "offset", "segment", "DIS segment identifier offset", "0"))},
#else
    //{'c', OptionPtr(new Option<const char*>(controllerPath, "path", "ctrl", "NVM controller device path"))},
#endif
        {'N', OptionPtr(new Option<const char*>(node_sequence, "path", "node_sequence", "node sequence input file"))},
        {'B', OptionPtr(new Option<const char*>(batch_size, "path", "batch_size", "batch size(number of node in one batch)"))},

        {'r', OptionPtr(new Option<bool>(graph_reorganize, "bool", "graph_reorganize", "1 to enable graph reorganize", "false"))},
        {'D', OptionPtr(new Option<const char*>(graph_reorg_node_map_path, "path", "graph_reorg_node_map_path", "node map file path"))},

        {'R', OptionPtr(new Option<bool>(neighbor_feature, "bool", "neighbor_feature", "1 to enable 1-hop neighbor feature", "false"))},
        {'I', OptionPtr(new Option<const char*>(neighbor_feature_node_list_path, "path", "neighbor_feature_node_list_path", "node list for whole feature"))},
        {'F', OptionPtr(new Option<const char*>(neighbor_len_path, "path", "neighbor_len_path", "number of 1-hop neighbor for nodes"))},

        {'g', OptionPtr(new Option<uint32_t>(cudaDevice, "number", "gpu", "specify CUDA device", "0"))},
        {'K', OptionPtr(new Option<const char*>(libnvmName, "path", "libnvmName", "/dev/libnvm0 or /dev/libnvm1"))},
        //{'i', OptionPtr(new Option<uint32_t>(nvmNamespace, "identifier", "namespace", "NVM namespace identifier", "1"))},
        //{'B', OptionPtr(new Option<bool>(doubleBuffered, "bool", "double-buffer", "double buffer disk reads", "false"))},
        //{'r', OptionPtr(new Option<bool>(stats, "bool", "stats", "print statistics", "false"))},
        {'P', OptionPtr(new Range(pageSize, 1, (uint64_t)std::numeric_limits<uint64_t>::max, "page_size", "size of page in cache", "4096"))},
        {'b', OptionPtr(new Range(blkSize, 1, (uint64_t)std::numeric_limits<uint64_t>::max, "blk_size", "CUDA thread block size", "128"))},
        {'d', OptionPtr(new Range(queueDepth, 2, 65536, "queue_depth", "queue depth per queue", "16"))},
        {'q', OptionPtr(new Range(numQueues, 1, 65536, "num_queues", "number of queues per controller", "1"))},
        {'S', OptionPtr(new Range(ssdtype, 0, 2, "ssd", "type of SSD to use 0->Samsung, 1->Intel", "0"))},
        {'E', OptionPtr(new Range(num_page, 0, (uint64_t)std::numeric_limits<uint64_t>::max(), "num_page", "num of page in SSD", "0"))},
        {'C', OptionPtr(new Range(cache_size, 0, (uint64_t)std::numeric_limits<uint64_t>::max(), "cache_size", "cache_size:(MB)", "1"))},

    };

    string optionString;
    vector<option> options;
    createLongOptions(options, optionString, parsers);

    int index;
    int option;
    OptionMap::iterator parser;

    while ((option = getopt_long(argc, argv, optionString.c_str(), &options[0], &index)) != -1) {
        switch (option) {
            case '?':
                throw string("Unknown option: `") + argv[optind - 1] + string("'");

            case ':':
                throw string("Missing argument for option `") + argv[optind - 1] + string("'");

            case 'h':
                throw helpString(argv[0], parsers);

            default:
                parser = parsers.find(option);
                if (parser == parsers.end()) {
                    throw string("Unknown option: `") + argv[optind - 1] + string("'");
                }
                parser->second->parseArgument(argv[optind - 1], optarg);
                break;
        }
    }
    /*
    #ifdef __DIS_CLUSTER__
        if (blockDevicePath == nullptr && controllerId == 0)
        {
            throw string("No block device or NVM controller specified");
        }
        else if (blockDevicePath != nullptr && controllerId != 0)
        {
            throw string("Either block device or NVM controller must be specified, not both!");
        }
    #else
        if (blockDevicePath == nullptr && controllerPath == nullptr)
        {
            throw string("No block device or NVM controller specified");
        }
        else if (blockDevicePath != nullptr && controllerPath != nullptr)
        {
            throw string("Either block device or NVM controller must be specified, not both!");
        }
    #endif

        if (blockDevicePath != nullptr && doubleBuffered)
        {
            throw string("Double buffered reading from block device is not supported");
        }
    */
    verifyCudaDevice(cudaDevice);
    // verifyNumberOfThreads(numThreads);

    setBDF(*this);
}

Settings::Settings() {
    cudaDevice = 0;
    cudaDeviceId = 0;
    blockDevicePath = nullptr;
    controllerPath = nullptr;
    controllerId = 0;
    adapter = 0;
    segmentId = 0;
    nvmNamespace = 1;
    doubleBuffered = false;
    numReqs = 1;
    numPages = 1024;
    startBlock = 0;
    stats = false;
    output = nullptr;
    numThreads = 64;
    blkSize = 64;
    domain = 0;
    bus = 0;
    devfn = 0;

    pageSize = 4096;  // 默认设置了page size
    // pageSize = 16384;
    numBlks = 2097152;
    random = true;
    accessType = READ;
    ratio = 100;
    input = nullptr;
    ssdtype = 0;

    libnvmName = nullptr;
    cache_size = 1;
    queueDepth = 16;
    numQueues = 1;
    graph_reorganize = false;
    neighbor_feature = false;
}

#endif

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <regex>
#include <fstream>
#include <string>
#include <random>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <tclap/CmdLine.h>
#include <hepnos.hpp>
#ifdef ONLY_TEST_CLASSES
#include "_test_.hpp"
#include "_test_macro_.hpp"
#else
#include "hepnos-nova-classes/_all_.hpp"
#include "hepnos-nova-classes/_macro_.hpp"
#endif
#include "DummyProduct.hpp"

static int                       g_size;
static int                       g_rank;
static std::string               g_protocol;
static std::string               g_connection_file;
static std::string               g_margo_file;
static bool                      g_compare;
static std::string               g_input_dataset;
static std::string               g_product_label;
static spdlog::level::level_enum g_logging_level;
static unsigned                  g_num_threads;
static std::vector<std::string>  g_product_names;
static bool                      g_preload_products;
static std::pair<double,double>  g_wait_range;
static std::unordered_map<
        std::string,
        std::function<void(const hepnos::Event&, const hepnos::ProductCache&)>>
                                 g_load_product_fn;
static std::unordered_map<
        std::string,
        std::function<void(hepnos::ParallelEventProcessor&)>>
                                 g_preload_fn;
static std::mt19937              g_mte;
static hepnos::ParallelEventProcessorOptions
                                 g_pep_options;
static bool                      g_disable_stats;

static void parse_arguments(int argc, char** argv);
static std::pair<double,double> parse_wait_range(const std::string&);
static std::string check_file_exists(const std::string& filename);
static void prepare_product_loading_functions();
static void prepare_preloading_functions();
static void run_benchmark();
template<typename Ostream>
static Ostream& operator<<(Ostream& os, const hepnos::ParallelEventProcessorStatistics& stats);

int main(int argc, char** argv) {

    int provided, required = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    std::stringstream str_format;
    str_format << "[" << std::setw(6) << std::setfill('0') << g_rank << "|" << g_size
               << "] [%H:%M:%S.%F] [%n] [%^%l%$] %v";
    spdlog::set_pattern(str_format.str());

    parse_arguments(argc, argv);

    spdlog::set_level(g_logging_level);

    if(provided != required && g_rank == 0) {
        spdlog::warn("MPI doesn't provider MPI_THREAD_MULTIPLE");
    }

    spdlog::trace("connection file: {}", g_connection_file);
    spdlog::trace("input dataset: {}", g_input_dataset);
    spdlog::trace("product label: {}", g_product_label);
    spdlog::trace("num threads: {}", g_num_threads);
    spdlog::trace("product names: {}", g_product_names.size());
    spdlog::trace("wait range: {},{}", g_wait_range.first, g_wait_range.second);

    prepare_product_loading_functions();
    if(g_preload_products) {
        prepare_preloading_functions();
    }

    if(g_rank == 0) {
        for(auto& p : g_product_names) {
            if(g_load_product_fn.count(p) == 0) {
                spdlog::critical("Unknown product name {}", p);
                MPI_Abort(MPI_COMM_WORLD, -1);
                exit(-1);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    spdlog::trace("Initializing RNG");
    g_mte = std::mt19937(g_rank);

    run_benchmark();

    MPI_Finalize();
    return 0;
}

static void parse_arguments(int argc, char** argv) {
    try {
        TCLAP::CmdLine cmd("Benchmark HEPnOS Parallel Event Processor", ' ', "0.6");
        // mandatory arguments
        TCLAP::ValueArg<std::string> protocol("p", "protocol",
            "Mercury protocol", true, "", "string");
        TCLAP::ValueArg<std::string> clientFile("c", "connection",
            "YAML connection file for HEPnOS", true, "", "string");
        TCLAP::ValueArg<std::string> dataSetName("d", "dataset",
            "DataSet from which to load the data", true, "", "string");
        TCLAP::ValueArg<std::string> productLabel("l", "label",
            "Label to use when storing products", true, "", "string");
        // optional arguments
        TCLAP::ValueArg<std::string> margoFile("m", "margo-config",
            "Margo configuration file", false, "", "string");
        std::vector<std::string> allowed = {
            "trace", "debug", "info", "warning", "error", "critical", "off" };
        TCLAP::ValuesConstraint<std::string> allowedVals( allowed );
        TCLAP::ValueArg<std::string> loggingLevel("v", "verbose",
            "Logging output type (info, debug, critical)", false, "info",
            &allowedVals);
        TCLAP::ValueArg<unsigned> numThreads("t", "threads",
            "Number of threads to run processing work", false, 0, "int");
        TCLAP::MultiArg<std::string> productNames("n", "product-names",
            "Name of the products to load", false, "string");
        TCLAP::ValueArg<std::string> waitRange("r", "wait-range",
            "Waiting time interval in seconds (e.g. 1.34,3.56)", false, "0,0", "x,y");
        TCLAP::ValueArg<unsigned> inputBatchSize("i", "input-batch-size",
            "Input batch size for parallel event processor", false, 16, "int");
        TCLAP::ValueArg<unsigned> outputBatchSize("o", "output-batch-size",
            "Output batch size for parallel event processor", false, 16, "int");
        TCLAP::ValueArg<unsigned> cacheSize("s", "cache-size",
            "Prefetcher cache size for parallel event processor", false,
            std::numeric_limits<unsigned>::max(), "int");
        TCLAP::SwitchArg preloadProducts("", "preload",
            "Enable preloading products");
        TCLAP::SwitchArg disableStats("", "disable-stats",
            "Disable statistics collection");
        TCLAP::SwitchArg compare("", "compare", "Compare with and without preloading");
        TCLAP::SwitchArg noRDMA("", "no-rdma", "Use RPC instead of RDMA for event sharing");

        cmd.add(protocol);
        cmd.add(margoFile);
        cmd.add(clientFile);
        cmd.add(dataSetName);
        cmd.add(productLabel);
        cmd.add(loggingLevel);
        cmd.add(numThreads);
        cmd.add(productNames);
        cmd.add(waitRange);
        cmd.add(inputBatchSize);
        cmd.add(outputBatchSize);
        cmd.add(cacheSize);
        cmd.add(disableStats);
        cmd.add(preloadProducts);
        cmd.add(compare);
        cmd.add(noRDMA);

        cmd.parse(argc, argv);

        g_protocol          = protocol.getValue();
        g_margo_file        = margoFile.getValue();
        g_connection_file   = check_file_exists(clientFile.getValue());
        g_input_dataset     = dataSetName.getValue();
        g_product_label     = productLabel.getValue();
        g_logging_level     = spdlog::level::from_str(loggingLevel.getValue());
        g_num_threads       = numThreads.getValue();
        g_product_names     = productNames.getValue();
        g_preload_products  = preloadProducts.getValue();
        g_wait_range        = parse_wait_range(waitRange.getValue());
        g_pep_options.inputBatchSize  = inputBatchSize.getValue();
        g_pep_options.outputBatchSize = outputBatchSize.getValue();
        g_pep_options.cacheSize       = cacheSize.getValue();
        g_pep_options.use_rdma        = !noRDMA.getValue();
        g_disable_stats               = disableStats.getValue();
        g_compare                     = compare.getValue();
        if(g_compare) g_preload_products = true;

    } catch(TCLAP::ArgException &e) {
        if(g_rank == 0) {
            spdlog::critical("{} for command-line argument {}", e.error(), e.argId());
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(-1);
        }
    }
}

static std::pair<double,double> parse_wait_range(const std::string& s) {
    std::pair<double,double> range = { 0.0, 0.0 };
    std::regex rgx("^((0|([1-9][0-9]*))(\\.[0-9]+)?)(,((0|([1-9][0-9]*))(\\.[0-9]+)?))?$");
    // groups 1 and 6 will contain the two numbers
    std::smatch matches;

    if(std::regex_search(s, matches, rgx)) {
        range.first = atof(matches[1].str().c_str());
        if(matches[6].str().size() != 0) {
            range.second = atof(matches[6].str().c_str());
        } else {
            range.second = range.first;
        }
    } else {
        if(g_rank == 0) {
            spdlog::critical("Invalid wait range expression {} (should be \"x,y\" where x and y are floats)", s);
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(-1);
        }
    }
    if(range.second < range.first) {
        spdlog::critical("Invalid wait range expression {} ({} < {})",
                         s, range.second, range.first);
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }

    return range;
}

static std::string check_file_exists(const std::string& filename) {
    spdlog::trace("Checking if file {} exists", filename);
    std::ifstream ifs(filename);
    if(ifs.good()) return filename;
    else {
        spdlog::critical("File {} does not exist", filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
    return "";
}

static void prepare_product_loading_functions() {
    spdlog::trace("Preparing functions for loading producs");
#define X(__class__) \
    g_load_product_fn[#__class__] = [](const hepnos::Event& ev, const hepnos::ProductCache& cache) { \
        std::vector<__class__> product; \
        spdlog::trace("Loading product of type " #__class__); \
        if(!g_compare) { \
            if(!g_preload_products) { \
                if(!ev.load(g_product_label, product)) { \
                    spdlog::error("Could not load product of type " #__class__); \
                } \
            } else { \
                if(!ev.load(cache, g_product_label, product)) { \
                    spdlog::error("Could not load product of type " #__class__ " from cache"); \
                } \
            } \
        } else { \
            decltype(product) preloaded_product; \
            if(!ev.load(g_product_label, product)) { \
                spdlog::error("Could not load product of type " #__class__); \
            } \
            if(!ev.load(cache, g_product_label, preloaded_product)) { \
                spdlog::error("Could not load product of type " #__class__ " from cache"); \
            } \
            auto rn  = ev.subrun().run().number(); \
            auto srn = ev.subrun().number(); \
            auto evn = ev.number(); \
            if(preloaded_product.size() != product.size()) { \
                spdlog::error("[{},{},{}] product " #__class__ " size error ({} != {})", \
                        rn, srn, evn, preloaded_product.size(), product.size()); \
            } else { \
                if(std::memcmp(preloaded_product.data(), product.data(), product.size()*sizeof(__class__)) != 0) { \
                    spdlog::error("[{},{},{}] product " #__class__ " binary differs", rn, srn, evn); \
                } \
            } \
        } \
    };

    X(dummy_product)
    HEPNOS_FOREACH_NOVA_CLASS
#undef X
    spdlog::trace("Created functions for {} product types", g_load_product_fn.size());
}

static void prepare_preloading_functions() {
    spdlog::trace("Preparing functions for loading producs");
#define X(__class__) \
    g_preload_fn[#__class__] = [](hepnos::ParallelEventProcessor& pep) { \
        spdlog::trace("Setting preload for product of type " #__class__); \
        pep.preload<std::vector<__class__>>(g_product_label); \
    };

    X(dummy_product)
    HEPNOS_FOREACH_NOVA_CLASS
#undef X
    spdlog::trace("Created functions for {} product types", g_load_product_fn.size());
}

static void simulate_processing(const hepnos::Event& ev, const hepnos::ProductCache& cache) {
    spdlog::trace("Loading products");
    try {
        for(auto& p : g_product_names) {
            g_load_product_fn[p](ev, cache);
        }
    } catch(const hepnos::Exception& ex) {
        spdlog::critical(ex.what());
    }
    spdlog::trace("Simulating processing");
    double t_start = MPI_Wtime();
    double t_wait;
    if(g_wait_range.first == g_wait_range.second) {
        t_wait = g_wait_range.first;
    } else {
        std::uniform_real_distribution<double> dist(
            g_wait_range.first, g_wait_range.second);
        t_wait = dist(g_mte);
    }
    double t_now;
    do {
        t_now = MPI_Wtime();
    } while(t_now - t_start < t_wait);
}

static void run_benchmark() {

    double t_start, t_end;
    hepnos::DataStore datastore;
    try {
        spdlog::trace("Connecting to HEPnOS using file {}", g_connection_file);
        datastore = hepnos::DataStore::connect(g_protocol, g_connection_file, g_margo_file);
    } catch(const hepnos::Exception& ex) {
        spdlog::critical("Could not connect to HEPnOS service: {}", ex.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    {

        spdlog::trace("Creating AsyncEngine with {} threads", g_num_threads);
        hepnos::AsyncEngine async(datastore, g_num_threads);

        spdlog::trace("Creating ParallelEventProcessor");
        hepnos::ParallelEventProcessor pep(async, MPI_COMM_WORLD, g_pep_options);

        if(g_preload_products) {
            spdlog::trace("Setting preload flags");
            for(auto& p : g_product_names) {
                g_preload_fn[p](pep);
            }
        }
        spdlog::trace("Loading dataset");
        hepnos::DataSet dataset;
        try {
            dataset = datastore.root()[g_input_dataset];
        } catch(...) {}
        if(!dataset.valid() && g_rank == 0) {
            spdlog::critical("Invalid dataset {}", g_input_dataset);
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(-1);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        spdlog::trace("Calling processing function on dataset {}", g_input_dataset);

        hepnos::ParallelEventProcessorStatistics stats;
        hepnos::ParallelEventProcessorStatistics* stats_ptr = &stats;
        if(g_disable_stats)
            stats_ptr = nullptr;

        MPI_Barrier(MPI_COMM_WORLD);
        t_start = MPI_Wtime();
        pep.process(dataset, [](const hepnos::Event& ev, const hepnos::ProductCache& cache) {
            auto subrun = ev.subrun();
            auto run = subrun.run();
            spdlog::trace("Processing event {} from subrun {} from run {}",
                      ev.number(), subrun.number(), run.number());
            simulate_processing(ev, cache);
        }, stats_ptr);
        MPI_Barrier(MPI_COMM_WORLD);
        t_end = MPI_Wtime();

        if(!g_disable_stats)
            spdlog::info("Statistics: {}", stats);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(g_rank == 0)
        spdlog::info("Benchmark completed in {} seconds", t_end-t_start);
}

template<typename Ostream>
static Ostream& operator<<(Ostream& os, const hepnos::ParallelEventProcessorStatistics& stats) {
    os << "{ \"total_events_processed\" : " << stats.total_events_processed << ","
       << " \"local_events_processed\" : " << stats.local_events_processed << ","
       << " \"total_time\" : " << stats.total_time << ","
       << " \"acc_event_processing_time\" : " << stats.acc_event_processing_time << ","
       << " \"acc_product_loading_time\" : " << stats.acc_product_loading_time << ","
       << " \"processing_time_stats\" : " << stats.processing_time_stats << ","
       << " \"waiting_time_stats\" : " << stats.waiting_time_stats << "}";
    return os;
}

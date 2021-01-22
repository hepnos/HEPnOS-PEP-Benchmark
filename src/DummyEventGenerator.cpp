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
#include "hepnos-nova-classes/_all_.hpp"
#include "hepnos-nova-classes/_macro_.hpp"
#include "DummyProduct.hpp"

static int                       g_size;
static int                       g_rank;
static std::string               g_connection_file;
static std::string               g_output_dataset;
static std::string               g_product_label;
static spdlog::level::level_enum g_logging_level;
static std::vector<std::string>  g_product_names;
static std::unordered_map<
        std::string,
        std::function<void(hepnos::Event&)>>
                                 g_store_product_fn;
static std::mt19937              g_mte;
static unsigned                  g_dummy_payload;
static unsigned                  g_num_runs;
static unsigned                  g_num_subruns;
static unsigned                  g_num_events;

static void parse_arguments(int argc, char** argv);
static std::string check_file_exists(const std::string& filename);
static void prepare_product_storing_functions();
static void generate_data();

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    std::stringstream str_format;
    str_format << "[" << std::setw(6) << std::setfill('0') << g_rank << "|" << g_size
               << "] [%H:%M:%S.%F] [%n] [%^%l%$] %v";
    spdlog::set_pattern(str_format.str());

    parse_arguments(argc, argv);

    spdlog::set_level(g_logging_level);

    spdlog::trace("connection file: {}", g_connection_file);
    spdlog::trace("output dataset: {}", g_output_dataset);
    spdlog::trace("product label: {}", g_product_label);
    spdlog::trace("product names: {}", g_product_names.size());

    prepare_product_storing_functions();

    spdlog::trace("Initializing RNG");
    g_mte = std::mt19937(g_rank);

    generate_data();

    MPI_Finalize();
    return 0;
}

static void parse_arguments(int argc, char** argv) {
    try {
        TCLAP::CmdLine cmd("Benchmark HEPnOS Parallel Event Processor", ' ', "0.1");
        // mandatory arguments
        TCLAP::ValueArg<std::string> clientFile("c", "connection",
            "YAML connection file for HEPnOS", true, "", "string");
        TCLAP::ValueArg<std::string> dataSetName("d", "dataset",
            "DataSet to which to store the data", true, "", "string");
        TCLAP::ValueArg<std::string> productLabel("l", "label",
            "Label to use when storing products", true, "", "string");
        // optional arguments
        std::vector<std::string> allowed = {
            "trace", "debug", "info", "warning", "error", "critical", "off" };
        TCLAP::ValuesConstraint<std::string> allowedVals( allowed );
        TCLAP::ValueArg<std::string> loggingLevel("v", "verbose",
            "Logging output type (info, debug, critical)", false, "info",
            &allowedVals);
        TCLAP::MultiArg<std::string> productNames("n", "product-names",
            "Name of the products to load", false, "string");
        TCLAP::ValueArg<unsigned> numRuns("r", "runs",
            "Number of runs to create", false, 8, "int");
        TCLAP::ValueArg<unsigned> numSubRuns("s", "subruns-per-run",
            "Number of subruns per run to create", false, 8, "int");
        TCLAP::ValueArg<unsigned> numEvents("e", "events-per-subrun",
            "Number of events per subrun to create", false, 8, "int");
        TCLAP::ValueArg<unsigned> payload("p", "dummy-payload",
            "Size of dummy products if provided", false, 8, "int");

        cmd.add(clientFile);
        cmd.add(dataSetName);
        cmd.add(productLabel);
        cmd.add(loggingLevel);
        cmd.add(productNames);
        cmd.add(numRuns);
        cmd.add(numSubRuns);
        cmd.add(numEvents);
        cmd.add(payload);

        cmd.parse(argc, argv);

        g_connection_file   = check_file_exists(clientFile.getValue());
        g_output_dataset    = dataSetName.getValue();
        g_product_label     = productLabel.getValue();
        g_logging_level     = spdlog::level::from_str(loggingLevel.getValue());
        g_product_names     = productNames.getValue();
        g_num_runs          = numRuns.getValue();
        g_num_subruns       = numSubRuns.getValue();
        g_num_events        = numEvents.getValue();
        g_dummy_payload     = payload.getValue();

    } catch(TCLAP::ArgException &e) {
        if(g_rank == 0) {
            spdlog::critical("{} for command-line argument {}", e.error(), e.argId());
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(-1);
        }
    }
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

static void prepare_product_storing_functions() {
    spdlog::trace("Preparing functions for loading producs");
#define X(__class__) \
    g_store_product_fn[#__class__] = [](hepnos::Event& ev) { \
        __class__ product; \
        ev.store(g_product_label, product); \
    };

    HEPNOS_FOREACH_NOVA_CLASS
#undef X
    g_store_product_fn["dummy_product"] = [](hepnos::Event& ev) {
        dummy_product product;
        product.data.resize(g_dummy_payload);
        ev.store(g_product_label, product);
    };
    spdlog::trace("Created functions for {} product types", g_store_product_fn.size());
}

static void generate_data() {

    hepnos::DataStore datastore;
    try {
        spdlog::trace("Connecting to HEPnOS using file {}", g_connection_file);
        datastore = hepnos::DataStore::connect(g_connection_file);
    } catch(const hepnos::Exception& ex) {
        spdlog::critical("Could not connect to HEPnOS service: {}", ex.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    spdlog::trace("creating dataset");
    hepnos::DataSet dataset;
    try {
        dataset = datastore.root().createDataSet(g_output_dataset);
    } catch(...) {}
    if(!dataset.valid() && g_rank == 0) {
        spdlog::critical("Could not create dataset {}", g_output_dataset);
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    spdlog::trace("Creating data");
    for(unsigned i = 0; i < g_num_runs; i++) {
        auto run = dataset.createRun(i);
        for(unsigned j = 0; j < g_num_subruns; j++) {
            auto subrun = run.createSubRun(j);
            for(unsigned k = 0; k < g_num_events; k++) {
                auto event = subrun.createEvent(k);
                for(auto& name : g_product_names) {
                    g_store_product_fn[name](event);
                }
            }
        }
    }
}

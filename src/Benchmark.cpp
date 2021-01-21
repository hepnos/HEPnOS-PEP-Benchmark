#include <mpi.h>
#include <iostream>
#include <sstream>
#include <regex>
#include <fstream>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <tclap/CmdLine.h>
#include <hepnos.hpp>
#include "hepnos-nova-classes/_all_.hpp"
#include "hepnos-nova-classes/_macro_.hpp"

static int                       g_size;
static int                       g_rank;
static std::string               g_connection_file;
static std::string               g_input_dataset;
static std::string               g_product_label;
static spdlog::level::level_enum g_logging_level;
static unsigned                  g_num_threads;
static std::vector<std::string>  g_product_names;
static std::pair<double,double>  g_wait_range;

static void parse_arguments(int argc, char** argv);
static std::pair<double,double> parse_wait_range(const std::string&);
static std::string check_file_exists(const std::string& filename);
static void run_benchmark();

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

    spdlog::debug("connection file: {}", g_connection_file);
    spdlog::debug("input dataset: {}", g_input_dataset);
    spdlog::debug("product label: {}", g_product_label);
    spdlog::debug("num threads: {}", g_num_threads);
    spdlog::debug("product names: {}", g_product_names.size());
    spdlog::debug("wait range: {},{}", g_wait_range.first, g_wait_range.second);

    run_benchmark();

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
            "DataSet from which to load the data", true, "", "string");
        TCLAP::ValueArg<std::string> productLabel("l", "label",
            "Label to use when storing products", true, "", "string");
        // optional arguments
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

        cmd.add(clientFile);
        cmd.add(dataSetName);
        cmd.add(productLabel);
        cmd.add(loggingLevel);
        cmd.add(numThreads);
        cmd.add(productNames);
        cmd.add(waitRange);

        cmd.parse(argc, argv);

        g_connection_file = check_file_exists(clientFile.getValue());
        g_input_dataset   = dataSetName.getValue();
        g_product_label   = productLabel.getValue();
        g_logging_level   = spdlog::level::from_str(loggingLevel.getValue());
        g_num_threads     = numThreads.getValue();
        g_product_names   = productNames.getValue();
        g_wait_range      = parse_wait_range(waitRange.getValue());

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

    return range;
}

static std::string check_file_exists(const std::string& filename) {
    std::ifstream ifs(filename);
    if(ifs.good()) return filename;
    else {
        spdlog::critical("File {} does not exist", filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
    return "";
}

static void run_benchmark() {
    // Initialize HEPnOS
    hepnos::DataStore datastore;
    try {
        spdlog::info("Connecting to HEPnOS using file {}", g_connection_file);
        datastore = hepnos::DataStore::connect(g_connection_file);
    } catch(const hepnos::Exception& ex) {
        spdlog::critical("Could not connect to HEPnOS service: {}", ex.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

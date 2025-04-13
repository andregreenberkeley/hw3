#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"

#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    std::string test_prefix = "test";
    if (run_type == "test" && argc >= 4) {
        test_prefix = std::string(argv[3]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);

    // Load factor of 0.5
    size_t hash_table_size = n_kmers * (1.0 / 0.5);
    // Initialize our distributed hash map
    HashMap hashmap(hash_table_size);

    if (run_type == "verbose") {
        BUtil::print("Rank %d: Initializing distributed hash table with local size %d (global size %d) for %d kmers.\n", 
                    upcxx::rank_me(), hashmap.size(), hashmap.global_table_size(), n_kmers);
    }

    // Each rank reads its portion of kmers
    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Rank %d: Finished reading %zu kmers.\n", upcxx::rank_me(), kmers.size());
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Each rank keeps track of its local start nodes
    std::vector<kmer_pair> start_nodes;

    // Insert kmers into the distributed hash map
    for (auto& kmer : kmers) {
        bool success = hashmap.insert(kmer);
        if (!success) {
            throw std::runtime_error("Error: HashMap is full!");
        }

        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }
    
    auto end_insert = std::chrono::high_resolution_clock::now();
    
    // Make sure all inserts are complete before we start assembly
    upcxx::barrier();

    if (run_type == "verbose") {
        double insert_time = std::chrono::duration<double>(end_insert - start).count();
        BUtil::print("Rank %d: Inserted %zu kmers in %lf seconds.\n", 
                     upcxx::rank_me(), kmers.size(), insert_time);
    }

    // A simpler approach: each rank just processes its own start nodes
    // This avoids complex communication patterns but still allows for parallelism
    std::vector<kmer_pair>& flattened_start_nodes = start_nodes;
    
    if (run_type == "verbose") {
        int local_start_count = start_nodes.size();
        int total_start_count = 0;
        upcxx::reduce_one(local_start_count, upcxx::experimental::op_add, 
                          0, upcxx::world()).wait();
        
        if (upcxx::rank_me() == 0) {
            total_start_count = local_start_count;
            BUtil::print("Total start nodes across all ranks: %d\n", total_start_count);
        }
    }
    
    if (run_type == "verbose") {
        BUtil::print("Rank %d: Collected %zu total start nodes across all ranks.\n", 
                    upcxx::rank_me(), flattened_start_nodes.size());
    }

    auto start_read = std::chrono::high_resolution_clock::now();

    // Each rank processes its own start nodes
    std::list<std::list<kmer_pair>> contigs;
    for (const auto& start_kmer : flattened_start_nodes) {
        std::list<kmer_pair> contig;
        contig.push_back(start_kmer);
        
        // Now we can follow the path through the distributed hash map
        while (contig.back().forwardExt() != 'F') {
            kmer_pair kmer;
            bool success = hashmap.find(contig.back().next_kmer(), kmer);
            
            if (!success) {
                // With a distributed hash map, this should not happen
                // unless there's something wrong with the input data
                throw std::runtime_error("Error: k-mer not found in hashmap.");
            }
            
            contig.push_back(kmer);
        }
        
        contigs.push_back(contig);
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    // Collect statistics across all ranks
    int local_contigs = contigs.size();
    int total_contigs = 0;
    
    // Perform reduction to rank 0
    upcxx::reduce_one(local_contigs, upcxx::experimental::op_add, 
                      0, upcxx::world()).wait();
    
    // On rank 0, capture the reduced result
    if (upcxx::rank_me() == 0) {
        total_contigs = local_contigs;
    }
    
    int local_kmers = numKmers;
    int total_kmers = 0;
    
    // Perform reduction to rank 0
    upcxx::reduce_one(local_kmers, upcxx::experimental::op_add, 
                      0, upcxx::world()).wait();
                      
    // On rank 0, capture the reduced result
    if (upcxx::rank_me() == 0) {
        total_kmers = local_kmers;
    }
                                        
    // Make results available to all ranks
    if (upcxx::rank_me() == 0) {
        if (run_type != "test") {
            BUtil::print("Assembled %d contigs with %d total k-mers in %lf seconds.\n", 
                        total_contigs, total_kmers, total.count());
        }
    }

    if (run_type == "verbose") {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, 
               flattened_start_nodes.size(),
               read.count(), insert.count(), total.count());
    }

    if (run_type == "test") {
        // Each rank writes its portion of the contigs
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}

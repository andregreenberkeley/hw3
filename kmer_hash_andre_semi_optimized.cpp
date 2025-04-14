#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "butil.hpp"

// Cache structure for contig assembly
struct LookupCache {
    std::vector<std::pair<pkmer_t, kmer_pair>> entries;
    size_t capacity;
    size_t mask;
    
    LookupCache(size_t size) {
        // Find next power of 2
        capacity = 1;
        while (capacity < size) capacity <<= 1;
        mask = capacity - 1;
        entries.resize(capacity);
        clear();
    }
    
    void clear() {
        for (auto& entry : entries) {
            entry.second = kmer_pair();
        }
    }
    
    bool find(const pkmer_t& key, kmer_pair& value) {
        size_t idx = key.hash() & mask;
        if (entries[idx].first == key && entries[idx].second.kmer == key) {
            value = entries[idx].second;
            return true;
        }
        return false;
    }
    
    void insert(const pkmer_t& key, const kmer_pair& value) {
        size_t idx = key.hash() & mask;
        entries[idx] = {key, value};
    }
};

// Optimized contig assembly function
std::list<std::list<kmer_pair>> assemble_contigs(
    const std::vector<kmer_pair>& start_nodes,
    HashMap& hashmap,
    bool verbose) {
    
    std::list<std::list<kmer_pair>> contigs;
    const int max_steps = 100000; // Prevent infinite loops
    LookupCache cache(4096);  // Cache for recent lookups
    
    // Process contigs in batches
    size_t batch_size = 32;
    for (size_t i = 0; i < start_nodes.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, start_nodes.size());
        std::vector<std::list<kmer_pair>> batch_contigs(end - i);
        
        // Clear cache between batches
        cache.clear();
        
        // Process each start node in the batch
        for (size_t j = i; j < end; j++) {
            std::list<kmer_pair>& contig = batch_contigs[j - i];
            contig.push_back(start_nodes[j]);
            
            int steps = 0;
            while (contig.back().forwardExt() != 'F' && steps < max_steps) {
                pkmer_t next_key = contig.back().next_kmer();
                kmer_pair next_kmer;
                
                // Try cache first
                if (cache.find(next_key, next_kmer)) {
                    contig.push_back(next_kmer);
                } else {
                    // If not in cache, query the hashmap
                    bool success = hashmap.find(next_key, next_kmer);
                    if (!success) {
                        if (verbose) {
                            BUtil::print("Rank %d: k-mer not found in hashmap.\n", upcxx::rank_me());
                        }
                        break;
                    }
                    
                    // Add to cache and contig
                    cache.insert(next_key, next_kmer);
                    contig.push_back(next_kmer);
                }
                
                steps++;
                
                // Process pending operations occasionally
                if (steps % 64 == 0) {
                    upcxx::progress();
                }
            }
            
            if (steps >= max_steps && verbose) {
                BUtil::print("Rank %d: Assembly aborted; possible infinite loop.\n", upcxx::rank_me());
            }
        }
        
        // Add batch results to contigs
        for (auto& contig : batch_contigs) {
            if (!contig.empty()) {
                contigs.push_back(std::move(contig));
            }
        }
        
        // Process any pending communication
        upcxx::progress();
    }
    
    return contigs;
}

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
    bool verbose = (run_type == "verbose");

    // Use an optimized hash table size - 1.5x instead of 4x
    size_t hash_table_size = n_kmers * 1.5;
    HashMap hashmap(hash_table_size);

    if (verbose) {
        BUtil::print("Rank %d: Initializing hash table of size %d for %d kmers.\n",
                     upcxx::rank_me(), hash_table_size, n_kmers);
    }

    // Read kmers
    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (verbose) {
        BUtil::print("Rank %d: Finished reading %zu kmers.\n", upcxx::rank_me(), kmers.size());
    }

    // Lightweight barrier
    upcxx::barrier();

    auto start = std::chrono::high_resolution_clock::now();

    // Collect start nodes and insert kmers
    std::vector<kmer_pair> start_nodes;
    start_nodes.reserve(kmers.size() / 10);
    
    // Sort kmers by rank to improve locality and reduce communication
    std::sort(kmers.begin(), kmers.end(), 
              [](const kmer_pair& a, const kmer_pair& b) { 
                  return (a.hash() % upcxx::rank_n()) < (b.hash() % upcxx::rank_n()); 
              });
    
    // Insert kmers in batches
    for (auto& kmer : kmers) {
        hashmap.insert(kmer);
        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }
    
    // Ensure all insertions are complete
    hashmap.flush_insertions();
    
    auto end_insert = std::chrono::high_resolution_clock::now();

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (verbose) {
        BUtil::print("Rank %d: Finished inserting %zu kmers in %lf seconds.\n",
                     upcxx::rank_me(), kmers.size(), insert_time);
    }

    // Sort start nodes to improve locality during assembly
    std::sort(start_nodes.begin(), start_nodes.end(),
              [](const kmer_pair& a, const kmer_pair& b) {
                  return a.hash() < b.hash();
              });
    
    auto start_read = std::chrono::high_resolution_clock::now();

    // Assemble contigs using the optimized function
    std::list<std::list<kmer_pair>> contigs = assemble_contigs(start_nodes, hashmap, verbose);

    auto end_read = std::chrono::high_resolution_clock::now();
    
    // Synchronize at the end
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    double read_time = std::chrono::duration<double>(end_read - start_read).count();
    double total = std::chrono::duration<double>(end - start).count();

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Rank %d: Assembled in %lf total\n", upcxx::rank_me(), total);
    }

    if (verbose) {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(),
               read_time, insert_time, total);
    }

    if (run_type == "test") {
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}

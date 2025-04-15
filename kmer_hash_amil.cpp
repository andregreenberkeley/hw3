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
#include <deque>

#include "hash_map.hpp" // Use our new async hash map
#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "butil.hpp"

// Cache structure for contig assembly - unchanged
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

// Structure to represent the state of a contig assembly
struct ContigState {
    std::list<kmer_pair> contig;
    bool completed = false;
    bool waiting = false;
    pkmer_t next_key;
    int steps = 0;
    
    explicit ContigState(const kmer_pair& start_node) {
        contig.push_back(start_node);
    }
};

// Optimized asynchronous contig assembly function
std::list<std::list<kmer_pair>> assemble_contigs_optimized(
    const std::vector<kmer_pair>& start_nodes,
    HashMap& hashmap,
    bool verbose) {
    
    std::list<std::list<kmer_pair>> completed_contigs;
    const int max_steps = 100000; // Prevent infinite loops
    LookupCache cache(8192);  // Larger cache for better hit rate
    
    // Active contigs being processed
    std::deque<ContigState> active_contigs;
    
    // Initialize with all start nodes
    for (const auto& start_node : start_nodes) {
        active_contigs.emplace_back(start_node);
    }
    
    // Keep track of contigs that are waiting for lookup results
    int waiting_contigs = 0;
    const int max_waiting = 64; // Maximum number of concurrent lookups
    
    // Batch processing of contigs
    while (!active_contigs.empty()) {
        // Process a batch from the front of the queue
        const size_t batch_size = std::min<size_t>(64, active_contigs.size());
        
        // Prepare keys for batch lookup
        std::vector<pkmer_t> batch_keys;
        batch_keys.reserve(batch_size);
        
        // First pass: process contigs and collect keys for batch lookup
        for (size_t i = 0; i < batch_size && waiting_contigs < max_waiting; ++i) {
            ContigState& state = active_contigs[i];
            
            // Skip completed or already waiting contigs
            if (state.completed || state.waiting) continue;
            
            // Check if we've reached the end or max steps
            if (state.contig.back().forwardExt() == 'F' || state.steps >= max_steps) {
                state.completed = true;
                continue;
            }
            
            // Get the next k-mer to find
            pkmer_t next_key = state.contig.back().next_kmer();
            kmer_pair next_kmer;
            
            // Try cache first
            if (cache.find(next_key, next_kmer)) {
                state.contig.push_back(next_kmer);
                state.steps++;
            } else {
                // Not in cache, add to batch lookup
                state.next_key = next_key;
                state.waiting = true;
                batch_keys.push_back(next_key);
                waiting_contigs++;
            }
        }
        
        // Process any pending communication
        hashmap.progress();
        
        // If we have keys to look up, do a batch find
        if (!batch_keys.empty()) {
            // Callback to process results as they come in
            auto process_result = [&](const pkmer_t& key, bool found, const kmer_pair& value) {
                // Find the contig waiting for this result
                for (auto& state : active_contigs) {
                    if (state.waiting && state.next_key == key) {
                        if (found) {
                            // Found the k-mer, add to contig and cache
                            state.contig.push_back(value);
                            cache.insert(key, value);
                            state.steps++;
                        } else {
                            // K-mer not found, mark contig as completed
                            if (verbose) {
                                BUtil::print("Rank %d: k-mer not found in hashmap.\n", upcxx::rank_me());
                            }
                            state.completed = true;
                        }
                        
                        // No longer waiting for this contig
                        state.waiting = false;
                        waiting_contigs--;
                        break;
                    }
                }
            };
            
            // Perform batch lookup with callback
            hashmap.find_batch(batch_keys, process_result);
        }
        
        // Move completed contigs to results and remove from active list
        size_t initial_size = active_contigs.size();
        for (size_t i = 0; i < initial_size; ++i) {
            if (active_contigs.front().completed && !active_contigs.front().waiting) {
                // If contig has more than just the start node, add to results
                if (active_contigs.front().contig.size() > 1) {
                    completed_contigs.push_back(std::move(active_contigs.front().contig));
                }
                active_contigs.pop_front();
            } else if (!active_contigs.front().waiting) {
                // If not waiting, move to the back to be processed again
                active_contigs.push_back(std::move(active_contigs.front()));
                active_contigs.pop_front();
            } else {
                // If waiting, leave it in place
                auto temp = std::move(active_contigs.front());
                active_contigs.pop_front();
                active_contigs.push_back(std::move(temp));
            }
        }
        
        // Process any pending communication after batch
        hashmap.progress();
    }
    
    return completed_contigs;
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
    auto start_read = std::chrono::high_resolution_clock::now();
    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());
    auto end_read = std::chrono::high_resolution_clock::now();
    double read_file_time = std::chrono::duration<double>(end_read - start_read).count();

    if (verbose) {
        BUtil::print("Rank %d: Finished reading %zu kmers in %lf seconds.\n", 
                     upcxx::rank_me(), kmers.size(), read_file_time);
    }

    // Barrier to ensure all ranks have read their kmers
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
    
    // Insert kmers in batches with progress processing
    for (size_t i = 0; i < kmers.size(); i++) {
        auto& kmer = kmers[i];
        hashmap.insert(kmer);
        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
        
        // Process progress occasionally
        if (i % 1024 == 0) {
            hashmap.progress();
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
    
    auto start_assembly = std::chrono::high_resolution_clock::now();

    // Assemble contigs using the optimized function
    std::list<std::list<kmer_pair>> contigs = assemble_contigs_optimized(start_nodes, hashmap, verbose);

    auto end_assembly = std::chrono::high_resolution_clock::now();
    
    // Synchronize at the end
    upcxx::barrier();
    
    auto end = std::chrono::high_resolution_clock::now();

    double assembly_time = std::chrono::duration<double>(end_assembly - start_assembly).count();
    double total = std::chrono::duration<double>(end - start).count();

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Rank %d: Assembled in %lf total\n", upcxx::rank_me(), total);
    }

    if (verbose) {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf assembly, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(),
               assembly_time, insert_time, total);
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

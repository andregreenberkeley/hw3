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
#include <cstring> // For memcpy

#include "hash_map.hpp" 
#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "butil.hpp"

// More efficient cache structure with better hit rates
struct LookupCache {
    static const size_t ASSOCIATIVITY = 4; // Set associativity
    
    struct CacheEntry {
        pkmer_t key;
        kmer_pair value;
        bool valid;
        
        CacheEntry() : valid(false) {}
    };
    
    std::vector<std::vector<CacheEntry>> sets;
    size_t num_sets;
    size_t total_hits;
    size_t total_misses;
    
    LookupCache(size_t size) {
        // Calculate number of sets (each set has ASSOCIATIVITY entries)
        num_sets = size / ASSOCIATIVITY;
        sets.resize(num_sets);
        
        // Initialize all sets with ASSOCIATIVITY entries each
        for (auto& set : sets) {
            set.resize(ASSOCIATIVITY);
        }
        
        clear();
    }
    
    void clear() {
        for (auto& set : sets) {
            for (auto& entry : set) {
                entry.valid = false;
            }
        }
        total_hits = 0;
        total_misses = 0;
    }
    
    size_t get_set_index(const pkmer_t& key) const {
        // Use lower bits of hash for set index
        return key.hash() % num_sets;
    }
    
    bool find(const pkmer_t& key, kmer_pair& value) {
        size_t set_idx = get_set_index(key);
        auto& set = sets[set_idx];
        
        // Search within the set
        for (const auto& entry : set) {
            if (entry.valid && entry.key == key) {
                value = entry.value;
                total_hits++;
                return true;
            }
        }
        
        total_misses++;
        return false;
    }
    
    void insert(const pkmer_t& key, const kmer_pair& value) {
        size_t set_idx = get_set_index(key);
        auto& set = sets[set_idx];
        
        // First, check if key already exists
        for (auto& entry : set) {
            if (entry.valid && entry.key == key) {
                entry.value = value;
                return;
            }
        }
        
        // Find an invalid entry or evict the oldest (first) entry
        for (auto& entry : set) {
            if (!entry.valid) {
                entry.key = key;
                entry.value = value;
                entry.valid = true;
                return;
            }
        }
        
        // All entries valid, replace the first one (LRU approximation)
        set[0].key = key;
        set[0].value = value;
    }
    
    double hit_rate() const {
        if (total_hits + total_misses == 0) return 0.0;
        return static_cast<double>(total_hits) / (total_hits + total_misses);
    }
};

// Optimized contig assembly with prefetching and better memory access patterns
std::list<std::list<kmer_pair>> assemble_contigs_optimized(
    const std::vector<kmer_pair>& start_nodes,
    HashMap& hashmap,
    bool verbose) {
    
    std::list<std::list<kmer_pair>> contigs;
    LookupCache cache(32768);  // Use a much larger cache
    
    // Process multiple contigs simultaneously
    const size_t MAX_ACTIVE_CONTIGS = 64;
    const int max_steps = 100000; // Prevent infinite loops
    
    // Keep track of active contigs and their next keys to look up
    struct ContigState {
        std::list<kmer_pair> contig;
        int steps;
        bool completed;
        
        ContigState(const kmer_pair& start) 
            : steps(0), completed(false) {
            contig.push_back(start);
        }
    };
    
    // Process start nodes in chunks
    for (size_t start_idx = 0; start_idx < start_nodes.size(); start_idx += MAX_ACTIVE_CONTIGS) {
        std::vector<ContigState> active_contigs;
        
        // Initialize batch of contigs
        size_t end_idx = std::min(start_idx + MAX_ACTIVE_CONTIGS, start_nodes.size());
        for (size_t i = start_idx; i < end_idx; i++) {
            active_contigs.emplace_back(start_nodes[i]);
        }
        
        // Process until all contigs in batch are completed
        bool all_completed;
        do {
            // Gather keys to look up
            std::vector<pkmer_t> lookup_keys;
            std::vector<kmer_pair> lookup_values;
            std::vector<size_t> contig_indices;
            
            for (size_t i = 0; i < active_contigs.size(); i++) {
                auto& state = active_contigs[i];
                
                if (!state.completed && 
                    state.contig.back().forwardExt() != 'F' && 
                    state.steps < max_steps) {
                    
                    pkmer_t next_key = state.contig.back().next_kmer();
                    kmer_pair next_value;
                    
                    // Try cache first
                    if (cache.find(next_key, next_value)) {
                        state.contig.push_back(next_value);
                        state.steps++;
                    } else {
                        // Not in cache, add to batch lookup
                        lookup_keys.push_back(next_key);
                        lookup_values.push_back(kmer_pair()); // Placeholder
                        contig_indices.push_back(i);
                    }
                } else {
                    state.completed = true;
                }
            }
            
            // Process any pending operations
            hashmap.progress();
            
            // If we have keys to look up, do a batch find
            if (!lookup_keys.empty()) {
                // Perform batch lookup
                std::vector<bool> results = hashmap.find_batch(lookup_keys, lookup_values);
                
                // Process results
                for (size_t i = 0; i < results.size(); i++) {
                    size_t contig_idx = contig_indices[i];
                    
                    if (results[i]) {
                        // Found the k-mer, add to contig and cache
                        active_contigs[contig_idx].contig.push_back(lookup_values[i]);
                        active_contigs[contig_idx].steps++;
                        cache.insert(lookup_keys[i], lookup_values[i]);
                    } else {
                        // K-mer not found, mark contig as completed
                        if (verbose) {
                            BUtil::print("Rank %d: k-mer not found in hashmap.\n", upcxx::rank_me());
                        }
                        active_contigs[contig_idx].completed = true;
                    }
                }
            }
            
            // Check if all contigs are completed
            all_completed = true;
            for (const auto& state : active_contigs) {
                if (!state.completed) {
                    all_completed = false;
                    break;
                }
            }
            
        } while (!all_completed);
        
        // Add completed contigs to the result
        for (auto& state : active_contigs) {
            if (state.steps >= max_steps && verbose) {
                BUtil::print("Rank %d: Assembly aborted; possible infinite loop.\n", upcxx::rank_me());
            }
            
            if (state.contig.size() > 1) {
                contigs.push_back(std::move(state.contig));
            }
        }
        
        // Process any pending operations between batches
        hashmap.progress();
    }
    
    if (verbose) {
        BUtil::print("Rank %d: Cache hit rate: %.2f%%\n", 
                  upcxx::rank_me(), cache.hit_rate() * 100.0);
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

    // Use a smaller hash table to reduce memory usage and improve locality
    size_t hash_table_size = n_kmers * 1.2;  // 1.2x instead of 1.5x
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

    // Barrier for synchronization
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
    
    // Insert kmers in batches with aggressive prefetching
    const size_t INSERT_BATCH_SIZE = 1024;
    for (size_t batch_start = 0; batch_start < kmers.size(); batch_start += INSERT_BATCH_SIZE) {
        size_t batch_end = std::min(batch_start + INSERT_BATCH_SIZE, kmers.size());
        
        // Insert this batch
        for (size_t i = batch_start; i < batch_end; i++) {
            hashmap.insert(kmers[i]);
            if (kmers[i].backwardExt() == 'F') {
                start_nodes.push_back(kmers[i]);
            }
        }
        
        // Process progress to handle pending communication
        hashmap.progress();
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

    // Assemble contigs using the high-performance function
    std::list<std::list<kmer_pair>> contigs = assemble_contigs_optimized(start_nodes, hashmap, verbose);

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

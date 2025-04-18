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
#include <unordered_map>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "butil.hpp"

// Optimized contig assembly function with overlapped communication
std::list<std::list<kmer_pair>> assemble_contigs_overlapped(
    const std::vector<kmer_pair>& start_nodes,
    HashMap& hashmap,
    bool verbose) {
    
    std::list<std::list<kmer_pair>> contigs;
    const int max_steps = 100000; // Prevent infinite loops
    
    // Process contigs in batches
    size_t batch_size = 64;  // Increased batch size
    for (size_t i = 0; i < start_nodes.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, start_nodes.size());
        std::vector<std::list<kmer_pair>> batch_contigs(end - i);
        
        // Track find operations for each contig
        std::vector<std::vector<int>> pending_find_ids(end - i);
        std::vector<bool> contig_done(end - i, false);
        
        // Initialize each contig with its start node
        for (size_t j = i; j < end; j++) {
            batch_contigs[j - i].push_back(start_nodes[j]);
        }
        
        // Track how many contigs are still active
        int active_contigs = end - i;
        
        // Continue until all contigs in the batch are done
        while (active_contigs > 0) {
            // Issue find operations for all active contigs
            for (size_t j = i; j < end; j++) {
                size_t contig_idx = j - i;
                
                // Skip if this contig is already done
                if (contig_done[contig_idx]) continue;
                
                auto& contig = batch_contigs[contig_idx];
                
                // If no pending operations, issue a new one
                if (pending_find_ids[contig_idx].empty()) {
                    // Get the next k-mer to find
                    pkmer_t next_key = contig.back().next_kmer();
                    
                    // Issue async find
                    int find_id = hashmap.queue_find(next_key);
                    pending_find_ids[contig_idx].push_back(find_id);
                }
            }
            
            // Process pending operations
            hashmap.process_pending_finds();
            hashmap.process_pending_insertions();
            
            // Check for completed find operations
            for (size_t j = i; j < end; j++) {
                size_t contig_idx = j - i;
                
                // Skip if this contig is already done
                if (contig_done[contig_idx]) continue;
                
                auto& contig = batch_contigs[contig_idx];
                auto& find_ids = pending_find_ids[contig_idx];
                
                // Check all pending finds for this contig
                for (auto it = find_ids.begin(); it != find_ids.end(); ) {
                    int find_id = *it;
                    
                    if (hashmap.is_find_ready(find_id)) {
                        // Get the result
                        auto result_opt = hashmap.get_find_result(find_id);
                        if (result_opt.has_value()) {
                            auto [found, next_kmer] = result_opt.value();
                            
                            if (found) {
                                // Add to contig
                                contig.push_back(next_kmer);
                                
                                // Check if we're at the end of this contig
                                if (contig.back().forwardExt() == 'F' || 
                                    contig.size() >= max_steps) {
                                    contig_done[contig_idx] = true;
                                    active_contigs--;
                                }
                            } else {
                                // K-mer not found, contig ends here
                                if (verbose) {
                                    BUtil::print("Rank %d: k-mer not found in hashmap.\n", upcxx::rank_me());
                                }
                                contig_done[contig_idx] = true;
                                active_contigs--;
                            }
                        }
                        
                        // Remove this find_id from the list
                        it = find_ids.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            
            // Yield to make progress on communications
            upcxx::progress();
        }
        
        // Add batch results to contigs
        for (auto& contig : batch_contigs) {
            if (!contig.empty()) {
                contigs.push_back(std::move(contig));
            }
        }
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

    // Lightweight barrier to ensure all ranks have their kmers
    upcxx::barrier();

    auto start = std::chrono::high_resolution_clock::now();

    // Collect start nodes and track kmer insertions
    std::vector<kmer_pair> start_nodes;
    start_nodes.reserve(kmers.size() / 10);
    
    // Sort kmers by rank to improve locality and reduce communication
    std::sort(kmers.begin(), kmers.end(), 
              [](const kmer_pair& a, const kmer_pair& b) { 
                  return (a.hash() % upcxx::rank_n()) < (b.hash() % upcxx::rank_n()); 
              });
    
    // Bulk insert with overlapped communication
    const size_t progress_interval = 1000; // How often to process pending operations
    for (size_t i = 0; i < kmers.size(); i++) {
        auto& kmer = kmers[i];
        
        // Insert asynchronously
        hashmap.insert_async(kmer);
        
        // Collect start nodes
        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
        
        // Periodically process pending operations
        if (i % progress_interval == 0) {
            hashmap.process_pending_insertions();
        }
    }
    
    // Ensure all insertions are complete before assembly
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

    // Assemble contigs using the optimized overlapped function
    std::list<std::list<kmer_pair>> contigs = assemble_contigs_overlapped(start_nodes, hashmap, verbose);

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

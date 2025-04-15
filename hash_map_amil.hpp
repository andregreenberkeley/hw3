#pragma once
#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <map>
#include <algorithm>
#include <queue>

const size_t BATCH_SIZE = 1024;

struct HashMapPart {
    // Use raw arrays for better performance
    upcxx::global_ptr<kmer_pair> data;
    upcxx::global_ptr<int> used;
    size_t size;

    // Need an explicit constructor since we can't use default constructor
    HashMapPart() : data(nullptr), used(nullptr), size(0) {}
    
    // Buffer for batched operations
    std::vector<std::pair<int, kmer_pair>> insertion_buffer;
};

struct HashMap {
    upcxx::dist_object<HashMapPart> distributed_map;
    size_t my_size;
    size_t global_size;
    upcxx::atomic_domain<int> ad_int;
    
    HashMap(size_t size) 
        : distributed_map(HashMapPart()),
          ad_int({upcxx::atomic_op::compare_exchange, upcxx::atomic_op::load}) {
        
        // Initialize my part of the hash table
        my_size = (size + upcxx::rank_n() - 1) / upcxx::rank_n(); // Ceiling division
        
        auto local_data = upcxx::new_array<kmer_pair>(my_size);
        auto local_used = upcxx::new_array<int>(my_size);
        
        // Initialize usage flags
        for (size_t i = 0; i < my_size; i++) {
            local_used.local()[i] = 0;
        }
        
        // Set up the distributed map
        distributed_map->data = local_data;
        distributed_map->used = local_used;
        distributed_map->size = my_size;
        distributed_map->insertion_buffer.reserve(BATCH_SIZE);
        
        // Calculate total size - use non-blocking reduction but wait for result
        global_size = upcxx::reduce_all(my_size, upcxx::op_fast_add).wait();
    }
    
    ~HashMap() {
        // Clean up allocated memory
        if (distributed_map->data) upcxx::delete_array(distributed_map->data);
        if (distributed_map->used) upcxx::delete_array(distributed_map->used);
    }
    
    // Process any pending communication
    void progress() {
        upcxx::progress();
    }
    
    bool insert(const kmer_pair& kmer) {
        uint64_t hash = kmer.hash();
        int target_rank = hash % upcxx::rank_n();
        
        // Process communication occasionally
        if (distributed_map->insertion_buffer.size() % 64 == 0) {
            progress();
        }
        
        if (target_rank == upcxx::rank_me()) {
            // Local insertion
            uint64_t start_slot = (hash / upcxx::rank_n()) % my_size;
            
            // Linear probing
            for (uint64_t probe = 0; probe < my_size; probe++) {
                uint64_t slot = (start_slot + probe) % my_size;
                
                // Try to claim the slot atomically - use blocking operation
                int prev = ad_int.compare_exchange(
                    distributed_map->used + slot, 0, 1, 
                    std::memory_order_relaxed
                ).wait();
                
                if (prev == 0) {
                    // Slot was empty, write the k-mer
                    distributed_map->data.local()[slot] = kmer;
                    return true;
                }
            }
            return false; // Table is full
        } else {
            // Buffer for remote insertion
            distributed_map->insertion_buffer.push_back({target_rank, kmer});
            
            // Flush when buffer is full
            if (distributed_map->insertion_buffer.size() >= BATCH_SIZE) {
                flush_insertion_buffer();
            }
            return true;
        }
    }
    
    void flush_insertion_buffer() {
        auto& buffer = distributed_map->insertion_buffer;
        if (buffer.empty()) return;
        
        // Sort by target rank to optimize communication
        std::sort(buffer.begin(), buffer.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Group insertions by rank
        int current_rank = -1;
        std::vector<kmer_pair> current_batch;
        
        for (const auto& [rank, kmer] : buffer) {
            if (rank != current_rank) {
                // Send previous batch if exists
                if (!current_batch.empty()) {
                    send_batch_to_rank(current_rank, current_batch);
                    current_batch.clear();
                }
                current_rank = rank;
            }
            current_batch.push_back(kmer);
        }
        
        // Send the last batch
        if (!current_batch.empty()) {
            send_batch_to_rank(current_rank, current_batch);
        }
        
        buffer.clear();
    }
    
    void send_batch_to_rank(int rank, const std::vector<kmer_pair>& kmers) {
        // Use fire-and-forget RPC for better performance
        upcxx::rpc_ff(rank,
            [](upcxx::dist_object<HashMapPart>& dmap, std::vector<kmer_pair> batch) {
                for (const auto& kmer : batch) {
                    uint64_t hash = kmer.hash();
                    uint64_t start_slot = (hash / upcxx::rank_n()) % dmap->size;
                    
                    // Try to find an empty slot
                    for (uint64_t probe = 0; probe < dmap->size; probe++) {
                        uint64_t slot = (start_slot + probe) % dmap->size;
                        if (dmap->used.local()[slot] == 0) {
                            // Found empty slot
                            dmap->used.local()[slot] = 1;
                            dmap->data.local()[slot] = kmer;
                            break;
                        }
                    }
                }
            },
            distributed_map, kmers
        );
    }
    
    // Find a key with potential to overlap computation and communication
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        uint64_t hash = key_kmer.hash();
        int target_rank = hash % upcxx::rank_n();
        
        // Process any pending communication
        progress();
        
        if (target_rank == upcxx::rank_me()) {
            // Local lookup
            uint64_t start_slot = (hash / upcxx::rank_n()) % my_size;
            
            for (uint64_t probe = 0; probe < my_size; probe++) {
                uint64_t slot = (start_slot + probe) % my_size;
                
                // Check if slot is used
                if (distributed_map->used.local()[slot] == 0) {
                    return false; // Empty slot, k-mer not found
                }
                
                // Check if this is our k-mer
                const kmer_pair& current = distributed_map->data.local()[slot];
                if (current.kmer == key_kmer) {
                    val_kmer = current;
                    return true;
                }
            }
            return false; // Not found
        } else {
            // Remote lookup using RPC
            auto result = upcxx::rpc(target_rank,
                [](upcxx::dist_object<HashMapPart>& dmap, pkmer_t key) -> std::pair<bool, kmer_pair> {
                    uint64_t hash = key.hash();
                    uint64_t start_slot = (hash / upcxx::rank_n()) % dmap->size;
                    
                    for (uint64_t probe = 0; probe < dmap->size; probe++) {
                        uint64_t slot = (start_slot + probe) % dmap->size;
                        
                        if (dmap->used.local()[slot] == 0) {
                            return {false, kmer_pair()}; // Not found
                        }
                        
                        const kmer_pair& current = dmap->data.local()[slot];
                        if (current.kmer == key) {
                            return {true, current};
                        }
                    }
                    return {false, kmer_pair()}; // Not found
                },
                distributed_map, key_kmer
            ).wait();
            
            if (result.first) {
                val_kmer = result.second;
                return true;
            }
            return false;
        }
    }
    
    // Perform a batch of finds without waiting for each one
    // Use a callback to process results when they're ready
    template<typename Callback>
    void find_batch(const std::vector<pkmer_t>& keys, Callback callback) {
        // Group keys by target rank
        std::map<int, std::vector<pkmer_t>> grouped_keys;
        
        for (const auto& key : keys) {
            uint64_t hash = key.hash();
            int target_rank = hash % upcxx::rank_n();
            
            if (target_rank == upcxx::rank_me()) {
                // Process local lookups immediately
                kmer_pair value;
                bool found = find(key, value);
                callback(key, found, value);
            } else {
                // Group remote lookups by rank
                grouped_keys[target_rank].push_back(key);
            }
        }
        
        // Send batch RPCs for each group
        for (const auto& [rank, rank_keys] : grouped_keys) {
            // Fire off non-blocking RPC for this batch
            upcxx::rpc(rank,
                [](upcxx::dist_object<HashMapPart>& dmap, std::vector<pkmer_t> batch_keys) 
                    -> std::vector<std::pair<pkmer_t, std::pair<bool, kmer_pair>>> {
                    
                    std::vector<std::pair<pkmer_t, std::pair<bool, kmer_pair>>> results;
                    results.reserve(batch_keys.size());
                    
                    for (const auto& key : batch_keys) {
                        uint64_t hash = key.hash();
                        uint64_t start_slot = (hash / upcxx::rank_n()) % dmap->size;
                        bool found = false;
                        kmer_pair value;
                        
                        for (uint64_t probe = 0; probe < dmap->size; probe++) {
                            uint64_t slot = (start_slot + probe) % dmap->size;
                            
                            if (dmap->used.local()[slot] == 0) {
                                break; // Not found
                            }
                            
                            const kmer_pair& current = dmap->data.local()[slot];
                            if (current.kmer == key) {
                                found = true;
                                value = current;
                                break;
                            }
                        }
                        
                        results.push_back({key, {found, value}});
                    }
                    
                    return results;
                },
                distributed_map, rank_keys
            ).then([callback](std::vector<std::pair<pkmer_t, std::pair<bool, kmer_pair>>> results) {
                // Process each result from the batch
                for (const auto& [key, result] : results) {
                    callback(key, result.first, result.second);
                }
            });
        }
    }
    
    void flush_insertions() {
        flush_insertion_buffer();
        upcxx::barrier();
    }
    
    size_t size() const noexcept { return global_size; }
};

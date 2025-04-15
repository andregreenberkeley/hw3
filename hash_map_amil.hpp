#pragma once
#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <map>
#include <algorithm>
#include <cassert>

// Significantly increase batch size for better network utilization
const size_t BATCH_SIZE = 4096;

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
    
    // Use direct local pointers for faster access to local data
    kmer_pair* local_data;
    int* local_used;
    
    HashMap(size_t size) 
        : distributed_map(HashMapPart()),
          ad_int({upcxx::atomic_op::compare_exchange, upcxx::atomic_op::load}),
          local_data(nullptr),
          local_used(nullptr) {
        
        // Initialize my part of the hash table
        my_size = (size + upcxx::rank_n() - 1) / upcxx::rank_n(); // Ceiling division
        
        auto local_data_ptr = upcxx::new_array<kmer_pair>(my_size);
        auto local_used_ptr = upcxx::new_array<int>(my_size);
        
        // Save local pointers for direct access
        local_data = local_data_ptr.local();
        local_used = local_used_ptr.local();
        
        // Initialize usage flags
        for (size_t i = 0; i < my_size; i++) {
            local_used[i] = 0;
        }
        
        // Set up the distributed map
        distributed_map->data = local_data_ptr;
        distributed_map->used = local_used_ptr;
        distributed_map->size = my_size;
        distributed_map->insertion_buffer.reserve(BATCH_SIZE);
        
        // Calculate total size
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
    
    // Fast check if a slot is empty (without atomics)
    bool is_slot_empty(uint64_t slot) const {
        return local_used[slot] == 0;
    }
    
    // Fast local find without RPC
    bool find_local(const pkmer_t& key_kmer, kmer_pair& val_kmer) const {
        uint64_t hash = key_kmer.hash();
        uint64_t start_slot = (hash / upcxx::rank_n()) % my_size;
        
        for (uint64_t probe = 0; probe < my_size; probe++) {
            uint64_t slot = (start_slot + probe) % my_size;
            
            // Check if slot is used
            if (is_slot_empty(slot)) {
                return false; // Empty slot, k-mer not found
            }
            
            // Check if this is our k-mer (direct access to avoid indirection)
            const kmer_pair& current = local_data[slot];
            if (current.kmer == key_kmer) {
                val_kmer = current;
                return true;
            }
        }
        return false; // Not found
    }
    
    bool insert(const kmer_pair& kmer) {
        uint64_t hash = kmer.hash();
        int target_rank = hash % upcxx::rank_n();
        
        if (target_rank == upcxx::rank_me()) {
            // Local insertion - use direct pointers for better performance
            uint64_t start_slot = (hash / upcxx::rank_n()) % my_size;
            
            // Linear probing
            for (uint64_t probe = 0; probe < my_size; probe++) {
                uint64_t slot = (start_slot + probe) % my_size;
                
                // Check if slot appears empty (quick check without atomic)
                if (is_slot_empty(slot)) {
                    // Try to claim the slot atomically
                    int prev = ad_int.compare_exchange(
                        distributed_map->used + slot, 0, 1, 
                        std::memory_order_relaxed
                    ).wait();
                    
                    if (prev == 0) {
                        // Slot was empty, write the k-mer directly
                        local_data[slot] = kmer;
                        return true;
                    }
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
        current_batch.reserve(BATCH_SIZE);
        
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
        // Use fire-and-forget RPC for speed
        upcxx::rpc_ff(rank,
            [](upcxx::dist_object<HashMapPart>& dmap, std::vector<kmer_pair> batch) {
                // Get direct pointers to local data for better performance
                kmer_pair* local_data = dmap->data.local();
                int* local_used = dmap->used.local();
                size_t size = dmap->size;
                
                for (const auto& kmer : batch) {
                    uint64_t hash = kmer.hash();
                    uint64_t start_slot = (hash / upcxx::rank_n()) % size;
                    
                    // Try to find an empty slot
                    for (uint64_t probe = 0; probe < size; probe++) {
                        uint64_t slot = (start_slot + probe) % size;
                        if (local_used[slot] == 0) {
                            // Found empty slot, use memory fence for correctness
                            __sync_synchronize(); // Memory barrier
                            local_used[slot] = 1;
                            local_data[slot] = kmer;
                            break;
                        }
                    }
                }
            },
            distributed_map, kmers
        );
    }
    
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        uint64_t hash = key_kmer.hash();
        int target_rank = hash % upcxx::rank_n();
        
        // Process any pending communication
        progress();
        
        if (target_rank == upcxx::rank_me()) {
            // Local lookup - use the optimized version
            return find_local(key_kmer, val_kmer);
        } else {
            // Remote lookup using RPC
            auto result = upcxx::rpc(target_rank,
                [](upcxx::dist_object<HashMapPart>& dmap, pkmer_t key) -> std::pair<bool, kmer_pair> {
                    // Get direct local pointers
                    kmer_pair* local_data = dmap->data.local();
                    int* local_used = dmap->used.local();
                    size_t size = dmap->size;
                    
                    uint64_t hash = key.hash();
                    uint64_t start_slot = (hash / upcxx::rank_n()) % size;
                    
                    for (uint64_t probe = 0; probe < size; probe++) {
                        uint64_t slot = (start_slot + probe) % size;
                        
                        if (local_used[slot] == 0) {
                            return {false, kmer_pair()}; // Not found
                        }
                        
                        const kmer_pair& current = local_data[slot];
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
    
    // Batch find with prefetching
    std::vector<bool> find_batch(const std::vector<pkmer_t>& keys, std::vector<kmer_pair>& values) {
        assert(keys.size() == values.size());
        std::vector<bool> results(keys.size(), false);
        
        // First pass: do local lookups
        for (size_t i = 0; i < keys.size(); i++) {
            uint64_t hash = keys[i].hash();
            int target_rank = hash % upcxx::rank_n();
            
            if (target_rank == upcxx::rank_me()) {
                results[i] = find_local(keys[i], values[i]);
            }
        }
        
        // Second pass: group remote lookups by rank
        std::map<int, std::vector<std::pair<size_t, pkmer_t>>> remote_lookups;
        
        for (size_t i = 0; i < keys.size(); i++) {
            uint64_t hash = keys[i].hash();
            int target_rank = hash % upcxx::rank_n();
            
            if (target_rank != upcxx::rank_me()) {
                remote_lookups[target_rank].push_back({i, keys[i]});
            }
        }
        
        // Launch all remote lookups
        std::vector<upcxx::future<std::vector<std::pair<size_t, std::pair<bool, kmer_pair>>>>> futures;
        
        for (const auto& [rank, lookups] : remote_lookups) {
            futures.push_back(upcxx::rpc(
                rank,
                [](upcxx::dist_object<HashMapPart>& dmap, 
                   std::vector<std::pair<size_t, pkmer_t>> batch) 
                   -> std::vector<std::pair<size_t, std::pair<bool, kmer_pair>>> {
                    
                    std::vector<std::pair<size_t, std::pair<bool, kmer_pair>>> batch_results;
                    batch_results.reserve(batch.size());
                    
                    // Get direct pointers for better performance
                    kmer_pair* local_data = dmap->data.local();
                    int* local_used = dmap->used.local();
                    size_t size = dmap->size;
                    
                    for (const auto& [idx, key] : batch) {
                        uint64_t hash = key.hash();
                        uint64_t start_slot = (hash / upcxx::rank_n()) % size;
                        bool found = false;
                        kmer_pair value;
                        
                        for (uint64_t probe = 0; probe < size; probe++) {
                            uint64_t slot = (start_slot + probe) % size;
                            
                            if (local_used[slot] == 0) {
                                break; // Not found
                            }
                            
                            const kmer_pair& current = local_data[slot];
                            if (current.kmer == key) {
                                found = true;
                                value = current;
                                break;
                            }
                        }
                        
                        batch_results.push_back({idx, {found, value}});
                    }
                    
                    return batch_results;
                },
                distributed_map, lookups
            ));
        }
        
        // Wait for all futures and process results
        for (auto& future : futures) {
            auto batch_results = future.wait();
            for (const auto& [idx, result] : batch_results) {
                if (result.first) {
                    values[idx] = result.second;
                    results[idx] = true;
                }
            }
        }
        
        return results;
    }
    
    void flush_insertions() {
        flush_insertion_buffer();
        upcxx::barrier();
    }
    
    size_t size() const noexcept { return global_size; }
};

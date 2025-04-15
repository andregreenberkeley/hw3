#pragma once
#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <map>
#include <algorithm>
#include <queue>
#include <optional>

// Increased batch size for better amortization of communication costs
const size_t BATCH_SIZE = 2048;

struct HashMapPart {
    // Use raw arrays for better performance
    upcxx::global_ptr<kmer_pair> data;
    upcxx::global_ptr<int> used;
    size_t size;

    // Need an explicit constructor since we can't use default constructor
    HashMapPart() : data(nullptr), used(nullptr), size(0) {}
    
    // Reorganize as rank-based buffers for better communication overlap
    std::vector<std::vector<kmer_pair>> insertion_buffers;
};

// Pending find operation tracking
struct PendingFind {
    pkmer_t key;
    upcxx::future<std::pair<bool, kmer_pair>> future;
    
    PendingFind(const pkmer_t& k, upcxx::future<std::pair<bool, kmer_pair>> f)
        : key(k), future(std::move(f)) {}
};

struct HashMap {
    upcxx::dist_object<HashMapPart> distributed_map;
    size_t my_size;
    size_t global_size;
    upcxx::atomic_domain<int> ad_int;
    
    // Track pending operations
    std::vector<upcxx::future<>> pending_insertions;
    std::vector<PendingFind> pending_finds;
    
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
        
        // Initialize buffers for each rank
        distributed_map->insertion_buffers.resize(upcxx::rank_n());
        for (auto& buffer : distributed_map->insertion_buffers) {
            buffer.reserve(BATCH_SIZE);
        }
        
        // Calculate total size
        global_size = upcxx::reduce_all(my_size, upcxx::op_fast_add).wait();
        
        // Reserve space for pending operations
        pending_insertions.reserve(upcxx::rank_n());
        pending_finds.reserve(100);
    }
    
    ~HashMap() {
        // Wait for all pending operations to complete
        wait_all();
        
        // Clean up allocated memory
        if (distributed_map->data) upcxx::delete_array(distributed_map->data);
        if (distributed_map->used) upcxx::delete_array(distributed_map->used);
    }
    
    // Non-blocking insert with pending operations tracking
    bool insert_async(const kmer_pair& kmer) {
        uint64_t hash = kmer.hash();
        int target_rank = hash % upcxx::rank_n();
        
        if (target_rank == upcxx::rank_me()) {
            // Local insertion
            return insert_local(kmer);
        } else {
            // Add to appropriate buffer
            auto& buffer = distributed_map->insertion_buffers[target_rank];
            buffer.push_back(kmer);
            
            // Flush if buffer is full
            if (buffer.size() >= BATCH_SIZE) {
                auto future = flush_buffer_for_rank(target_rank);
                pending_insertions.push_back(std::move(future));
            }
            
            return true;
        }
    }
    
    // Local insertion helper
    bool insert_local(const kmer_pair& kmer) {
        uint64_t hash = kmer.hash();
        uint64_t start_slot = (hash / upcxx::rank_n()) % my_size;
        
        // Linear probing
        for (uint64_t probe = 0; probe < my_size; probe++) {
            uint64_t slot = (start_slot + probe) % my_size;
            
            // Try to claim the slot atomically
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
    }
    
    // Asynchronous flush for a specific rank
    upcxx::future<> flush_buffer_for_rank(int rank) {
        auto& buffer = distributed_map->insertion_buffers[rank];
        if (buffer.empty()) return upcxx::make_future();
        
        // Create a copy of the buffer to send
        std::vector<kmer_pair> batch_to_send = std::move(buffer);
        buffer = std::vector<kmer_pair>(); // Reset buffer
        buffer.reserve(BATCH_SIZE);
        
        // Send asynchronously and return future
        return upcxx::rpc(rank,
            [](upcxx::dist_object<HashMapPart>& dmap, std::vector<kmer_pair> batch) {
                for (const auto& kmer : batch) {
                    uint64_t hash = kmer.hash();
                    uint64_t start_slot = (hash / upcxx::rank_n()) % dmap->size;
                    
                    // Linear probing
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
            distributed_map, std::move(batch_to_send)
        );
    }
    
    // Process completed insertion operations
    void process_pending_insertions() {
        // Move completed futures out
        std::vector<upcxx::future<>> still_pending;
        still_pending.reserve(pending_insertions.size());
        
        for (auto& future : pending_insertions) {
            if (future.ready()) {
                // This future is complete, nothing more to do
            } else {
                still_pending.push_back(std::move(future));
            }
        }
        
        pending_insertions = std::move(still_pending);
        
        // Process progress to move communications forward
        upcxx::progress();
    }
    
    // Non-blocking find with future
    upcxx::future<std::pair<bool, kmer_pair>> find_async(const pkmer_t& key_kmer) {
        uint64_t hash = key_kmer.hash();
        int target_rank = hash % upcxx::rank_n();
        
        if (target_rank == upcxx::rank_me()) {
            // Local lookup - immediate result
            kmer_pair result;
            bool found = find_local(key_kmer, result);
            return upcxx::make_future(std::make_pair(found, result));
        } else {
            // Remote lookup using RPC
            return upcxx::rpc(target_rank,
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
            );
        }
    }
    
    // Local find helper
    bool find_local(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        uint64_t hash = key_kmer.hash();
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
    }
    
    // Queue a find operation and return an ID to track it
    int queue_find(const pkmer_t& key_kmer) {
        auto future = find_async(key_kmer);
        int id = pending_finds.size();
        pending_finds.emplace_back(key_kmer, std::move(future));
        return id;
    }
    
    // Check if a specific find is complete
    bool is_find_ready(int id) {
        if (id >= 0 && id < pending_finds.size()) {
            return pending_finds[id].future.ready();
        }
        return false;
    }
    
    // Get result of a pending find
    std::optional<std::pair<bool, kmer_pair>> get_find_result(int id) {
        if (id >= 0 && id < pending_finds.size() && pending_finds[id].future.ready()) {
            return pending_finds[id].future.wait();
        }
        return std::nullopt;
    }
    
    // Process pending finds that are ready
    void process_pending_finds() {
        // Process progress to move communications forward
        upcxx::progress();
    }
    
    // Flush all insertion buffers asynchronously
    void flush_insertions_async() {
        for (int rank = 0; rank < upcxx::rank_n(); rank++) {
            if (!distributed_map->insertion_buffers[rank].empty()) {
                auto future = flush_buffer_for_rank(rank);
                pending_insertions.push_back(std::move(future));
            }
        }
    }
    
    // Wait for all pending operations to complete
    void wait_all() {
        // Flush any remaining buffers
        flush_insertions_async();
        
        // Wait for all pending insertions
        for (auto& future : pending_insertions) {
            future.wait();
        }
        pending_insertions.clear();
        
        // Wait for all pending finds
        for (auto& find : pending_finds) {
            find.future.wait();
        }
        pending_finds.clear();
        
        // Global synchronization
        upcxx::barrier();
    }
    
    // Original API compatibility functions
    bool insert(const kmer_pair& kmer) {
        return insert_async(kmer);
    }
    
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        auto result = find_async(key_kmer).wait();
        if (result.first) {
            val_kmer = result.second;
            return true;
        }
        return false;
    }
    
    void flush_insertions() {
        flush_insertions_async();
        for (auto& future : pending_insertions) {
            future.wait();
        }
        pending_insertions.clear();
        upcxx::barrier();
    }
    
    size_t size() const noexcept { return global_size; }
};

#pragma once
#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <map>
#include <algorithm>
#include <queue>
#include <optional>

// Increased batch size for better communication amortization
const size_t BATCH_SIZE = 8192;  // Increased from 2048

struct HashMapPart {
    // Use raw arrays for better performance
    upcxx::global_ptr<kmer_pair> data;
    upcxx::global_ptr<int> used;
    size_t size;
    
    // Pre-computed mask for faster modulo operations
    uint64_t size_mask;

    // Need an explicit constructor since we can't use default constructor
    HashMapPart() : data(nullptr), used(nullptr), size(0), size_mask(0) {}

    // Reorganize as rank-based buffers for better communication overlap
    std::vector<std::vector<kmer_pair>> insertion_buffers;
    
    // Local cache to avoid RPC calls for frequently accessed kmers
    static constexpr size_t CACHE_SIZE = 1024;
    kmer_pair cache[CACHE_SIZE];
    uint64_t cache_keys[CACHE_SIZE];
    bool cache_valid[CACHE_SIZE];
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
    
    // Batch find operations for better network utilization
    std::vector<std::vector<pkmer_t>> find_buffers;
    static constexpr size_t FIND_BATCH_SIZE = 64;

    // Track slot usage for faster insertion
    uint64_t* slot_hints;

    HashMap(size_t size)
        : distributed_map(HashMapPart()),
          ad_int({upcxx::atomic_op::compare_exchange, upcxx::atomic_op::load}) {

        // Round up to power of 2 for fast modulo with mask
        size_t adjusted_size = 1;
        while (adjusted_size < size) adjusted_size *= 2;
        
        // Initialize my part of the hash table
        my_size = (adjusted_size + upcxx::rank_n() - 1) / upcxx::rank_n();
        
        // Ensure my_size is power of 2 for fast modulo
        size_t size_pow2 = 1;
        while (size_pow2 < my_size) size_pow2 *= 2;
        my_size = size_pow2;

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
        distributed_map->size_mask = my_size - 1; // For fast modulo with bitwise AND

        // Initialize buffers for each rank
        distributed_map->insertion_buffers.resize(upcxx::rank_n());
        for (auto& buffer : distributed_map->insertion_buffers) {
            buffer.reserve(BATCH_SIZE);
        }
        
        // Initialize find buffers for each rank
        find_buffers.resize(upcxx::rank_n());
        for (auto& buffer : find_buffers) {
            buffer.reserve(FIND_BATCH_SIZE);
        }

        // Initialize cache
        for (size_t i = 0; i < HashMapPart::CACHE_SIZE; i++) {
            distributed_map->cache_valid[i] = false;
        }

        // Calculate total size
        global_size = upcxx::reduce_all(my_size, upcxx::op_fast_add).wait();

        // Reserve space for pending operations
        pending_insertions.reserve(upcxx::rank_n() * 2);
        pending_finds.reserve(FIND_BATCH_SIZE * upcxx::rank_n());
        
        // Initialize slot hints for faster insertion
        slot_hints = new uint64_t[upcxx::rank_n()];
        for (int i = 0; i < upcxx::rank_n(); i++) {
            slot_hints[i] = 0;
        }
    }

    ~HashMap() {
        // Wait for all pending operations to complete
        wait_all();

        // Clean up allocated memory
        if (distributed_map->data) upcxx::delete_array(distributed_map->data);
        if (distributed_map->used) upcxx::delete_array(distributed_map->used);
        delete[] slot_hints;
    }

    // Fast hash function for better distribution
    inline uint64_t better_hash(uint64_t hash) const {
        // FNV-1a inspired mixing for better distribution
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccd;
        hash ^= hash >> 33;
        hash *= 0xc4ceb9fe1a85ec53;
        hash ^= hash >> 33;
        return hash;
    }

    // Non-blocking insert with pending operations tracking
    bool insert_async(const kmer_pair& kmer) {
        uint64_t hash = better_hash(kmer.hash());
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
                
                // Update the cache for frequently requested elements
                if (buffer.size() > 0) {
                    for (size_t i = 0; i < std::min(buffer.size(), size_t(8)); i++) {
                        size_t idx = buffer[i].hash() % HashMapPart::CACHE_SIZE;
                        distributed_map->cache[idx] = buffer[i];
                        distributed_map->cache_keys[idx] = buffer[i].kmer.hash();
                        distributed_map->cache_valid[idx] = true;
                    }
                }
            }

            return true;
        }
    }

    // Local insertion helper with optimized slot finding
    bool insert_local(const kmer_pair& kmer) {
        uint64_t hash = better_hash(kmer.hash());
        // Fast modulo with mask (works because my_size is power of 2)
        uint64_t start_slot = (hash / upcxx::rank_n()) & distributed_map->size_mask;
        
        // Use hint from previous successful insertion
        uint64_t hint = slot_hints[upcxx::rank_me()];
        uint64_t probe_start = (hint > 0) ? hint : start_slot;
        
        // First try starting from the hint
        for (uint64_t i = 0; i < 8; i++) {
            uint64_t slot = (probe_start + i) & distributed_map->size_mask;
            
            // Try to claim the slot atomically without waiting
            int prev = ad_int.compare_exchange(
                distributed_map->used + slot, 0, 1,
                std::memory_order_relaxed
            ).wait();

            if (prev == 0) {
                // Slot was empty, write the k-mer
                distributed_map->data.local()[slot] = kmer;
                // Update hint with this successful slot
                slot_hints[upcxx::rank_me()] = slot;
                return true;
            }
        }
        
        // Regular probing if hint didn't work
        for (uint64_t probe = 0; probe < my_size; probe++) {
            uint64_t slot = (start_slot + probe) & distributed_map->size_mask;

            // Skip slots we already tried with the hint
            if (probe < 8 && slot >= probe_start && slot < probe_start + 8) {
                continue;
            }

            // Use atomic operations to claim slot
            int prev = ad_int.compare_exchange(
                distributed_map->used + slot, 0, 1,
                std::memory_order_relaxed
            ).wait();

            if (prev == 0) {
                // Slot was empty, write the k-mer
                distributed_map->data.local()[slot] = kmer;
                // Update hint with this successful slot
                slot_hints[upcxx::rank_me()] = slot;
                return true;
            }
        }
        return false; // Table is full
    }

    // Asynchronous flush for a specific rank with SIMD optimizations
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
                // Sort by hash for better cache locality
                std::sort(batch.begin(), batch.end(), 
                    [](const kmer_pair& a, const kmer_pair& b) {
                        return a.hash() < b.hash();
                    });
                
                for (const auto& kmer : batch) {
                    uint64_t hash = kmer.hash();
                    // Fast modulo with mask
                    uint64_t start_slot = (hash / upcxx::rank_n()) & dmap->size_mask;

                    // Linear probing with prefetching
                    for (uint64_t probe = 0; probe < dmap->size; probe++) {
                        uint64_t slot = (start_slot + probe) & dmap->size_mask;
                        
                        // Prefetch next few slots for better cache performance
                        if (probe + 4 < dmap->size) {
                            __builtin_prefetch(&dmap->used.local()[(start_slot + probe + 4) & dmap->size_mask], 0, 1);
                        }
                        
                        if (dmap->used.local()[slot] == 0) {
                            // Found empty slot
                            dmap->used.local()[slot] = 1;
                            dmap->data.local()[slot] = kmer;
                            
                            // Update cache
                            size_t cache_idx = kmer.hash() % HashMapPart::CACHE_SIZE;
                            dmap->cache[cache_idx] = kmer;
                            dmap->cache_keys[cache_idx] = kmer.kmer.hash();
                            dmap->cache_valid[cache_idx] = true;
                            
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
        if (pending_insertions.empty()) return;
        
        // Move completed futures out
        std::vector<upcxx::future<>> still_pending;
        still_pending.reserve(pending_insertions.size());

        for (auto& future : pending_insertions) {
            if (future.is_ready()) {
                // This future is complete, nothing more to do
            } else {
                still_pending.push_back(std::move(future));
            }
        }

        pending_insertions = std::move(still_pending);

        // Process progress to move communications forward
        upcxx::progress();
    }

    // Check cache for find operations
    bool check_cache(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        uint64_t hash = key_kmer.hash();
        size_t cache_idx = hash % HashMapPart::CACHE_SIZE;
        
        if (distributed_map->cache_valid[cache_idx] && 
            distributed_map->cache_keys[cache_idx] == hash &&
            distributed_map->cache[cache_idx].kmer == key_kmer) {
            val_kmer = distributed_map->cache[cache_idx];
            return true;
        }
        return false;
    }

    // Non-blocking find with future, optimized with cache
    upcxx::future<std::pair<bool, kmer_pair>> find_async(const pkmer_t& key_kmer) {
        uint64_t hash = better_hash(key_kmer.hash());
        int target_rank = hash % upcxx::rank_n();
        
        // Check the cache first
        kmer_pair cached_value;
        if (check_cache(key_kmer, cached_value)) {
            return upcxx::make_future(std::make_pair(true, cached_value));
        }

        if (target_rank == upcxx::rank_me()) {
            // Local lookup - immediate result
            kmer_pair result;
            bool found = find_local(key_kmer, result);
            return upcxx::make_future(std::make_pair(found, result));
        } else {
            // Check if we should batch this find
            auto& buffer = find_buffers[target_rank];
            buffer.push_back(key_kmer);
            
            if (buffer.size() >= FIND_BATCH_SIZE) {
                // Perform batched find
                return batch_find_for_rank(target_rank, key_kmer);
            } else {
                // Remote lookup using RPC
                return upcxx::rpc(target_rank,
                    [](upcxx::dist_object<HashMapPart>& dmap, pkmer_t key) -> std::pair<bool, kmer_pair> {
                        uint64_t hash = key.hash();
                        // Fast modulo with mask
                        uint64_t start_slot = (hash / upcxx::rank_n()) & dmap->size_mask;

                        for (uint64_t probe = 0; probe < dmap->size; probe++) {
                            uint64_t slot = (start_slot + probe) & dmap->size_mask;
                            
                            // Prefetch next slots
                            if (probe + 4 < dmap->size) {
                                __builtin_prefetch(&dmap->used.local()[(start_slot + probe + 4) & dmap->size_mask], 0, 1);
                                __builtin_prefetch(&dmap->data.local()[(start_slot + probe + 4) & dmap->size_mask], 0, 1);
                            }

                            if (dmap->used.local()[slot] == 0) {
                                return {false, kmer_pair()}; // Not found
                            }

                            const kmer_pair& current = dmap->data.local()[slot];
                            if (current.kmer == key) {
                                // Update cache before returning
                                size_t cache_idx = key.hash() % HashMapPart::CACHE_SIZE;
                                dmap->cache[cache_idx] = current;
                                dmap->cache_keys[cache_idx] = key.hash();
                                dmap->cache_valid[cache_idx] = true;
                                
                                return {true, current};
                            }
                        }
                        return {false, kmer_pair()}; // Not found
                    },
                    distributed_map, key_kmer
                );
            }
        }
    }
    
    // Batch find operations for better network utilization
    upcxx::future<std::pair<bool, kmer_pair>> batch_find_for_rank(int target_rank, const pkmer_t& key_kmer) {
        auto& buffer = find_buffers[target_rank];
        if (buffer.empty()) return upcxx::make_future(std::make_pair(false, kmer_pair()));
        
        // Create a copy of the buffer to send
        std::vector<pkmer_t> batch_to_send = std::move(buffer);
        buffer = std::vector<pkmer_t>(); // Reset buffer
        buffer.reserve(FIND_BATCH_SIZE);
        
        // Send all keys in a batch, get back results for all
        auto batch_future = upcxx::rpc(target_rank,
            [](upcxx::dist_object<HashMapPart>& dmap, std::vector<pkmer_t> keys) -> std::vector<std::pair<pkmer_t, kmer_pair>> {
                std::vector<std::pair<pkmer_t, kmer_pair>> results;
                results.reserve(keys.size());
                
                // Sort keys by hash for better cache locality
                std::sort(keys.begin(), keys.end(), 
                    [](const pkmer_t& a, const pkmer_t& b) {
                        return a.hash() < b.hash();
                    });
                
                for (const auto& key : keys) {
                    uint64_t hash = key.hash();
                    // Fast modulo with mask
                    uint64_t start_slot = (hash / upcxx::rank_n()) & dmap->size_mask;
                    bool found = false;
                    
                    for (uint64_t probe = 0; probe < dmap->size; probe++) {
                        uint64_t slot = (start_slot + probe) & dmap->size_mask;
                        
                        // Prefetch
                        if (probe + 4 < dmap->size) {
                            __builtin_prefetch(&dmap->used.local()[(start_slot + probe + 4) & dmap->size_mask], 0, 1);
                            __builtin_prefetch(&dmap->data.local()[(start_slot + probe + 4) & dmap->size_mask], 0, 1);
                        }
                        
                        if (dmap->used.local()[slot] == 0) {
                            break; // Not found
                        }
                        
                        const kmer_pair& current = dmap->data.local()[slot];
                        if (current.kmer == key) {
                            results.emplace_back(key, current);
                            found = true;
                            
                            // Update cache
                            size_t cache_idx = key.hash() % HashMapPart::CACHE_SIZE;
                            dmap->cache[cache_idx] = current;
                            dmap->cache_keys[cache_idx] = key.hash();
                            dmap->cache_valid[cache_idx] = true;
                            
                            break;
                        }
                    }
                    
                    if (!found) {
                        results.emplace_back(key, kmer_pair());
                    }
                }
                
                return results;
            },
            distributed_map, std::move(batch_to_send)
        );
        
        // Extract the specific result for this key from the batch results
        return batch_future.then([key_kmer](std::vector<std::pair<pkmer_t, kmer_pair>> results) {
            for (const auto& result : results) {
                if (result.first == key_kmer) {
                    if (result.second.kmer == key_kmer) {
                        return std::make_pair(true, result.second);
                    } else {
                        return std::make_pair(false, kmer_pair());
                    }
                }
            }
            return std::make_pair(false, kmer_pair());
        });
    }

    // Local find helper with optimization
    bool find_local(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        // Check cache first
        if (check_cache(key_kmer, val_kmer)) {
            return true;
        }
        
        uint64_t hash = better_hash(key_kmer.hash());
        // Fast modulo with mask
        uint64_t start_slot = (hash / upcxx::rank_n()) & distributed_map->size_mask;

        for (uint64_t probe = 0; probe < my_size; probe++) {
            uint64_t slot = (start_slot + probe) & distributed_map->size_mask;
            
            // Prefetch next slots
            if (probe + 4 < my_size) {
                __builtin_prefetch(&distributed_map->used.local()[(start_slot + probe + 4) & distributed_map->size_mask], 0, 1);
                __builtin_prefetch(&distributed_map->data.local()[(start_slot + probe + 4) & distributed_map->size_mask], 0, 1);
            }

            // Check if slot is used
            if (distributed_map->used.local()[slot] == 0) {
                return false; // Empty slot, k-mer not found
            }

            // Check if this is our k-mer
            const kmer_pair& current = distributed_map->data.local()[slot];
            if (current.kmer == key_kmer) {
                val_kmer = current;
                
                // Update cache
                size_t cache_idx = key_kmer.hash() % HashMapPart::CACHE_SIZE;
                distributed_map->cache[cache_idx] = current;
                distributed_map->cache_keys[cache_idx] = key_kmer.hash();
                distributed_map->cache_valid[cache_idx] = true;
                
                return true;
            }
        }
        return false; // Not found
    }

    // Queue a find operation and return an ID to track it
    int queue_find(const pkmer_t& key_kmer) {
        // Check cache first
        kmer_pair val_kmer;
        if (check_cache(key_kmer, val_kmer)) {
            // Create a ready future with the cached result
            auto future = upcxx::make_future(std::make_pair(true, val_kmer));
            int id = pending_finds.size();
            pending_finds.emplace_back(key_kmer, std::move(future));
            return id;
        }
        
        auto future = find_async(key_kmer);
        int id = pending_finds.size();
        pending_finds.emplace_back(key_kmer, std::move(future));
        return id;
    }

    // Check if a specific find is complete
    bool is_find_ready(int id) {
        if (id >= 0 && id < pending_finds.size()) {
            return pending_finds[id].future.is_ready();
        }
        return false;
    }

    // Get result of a pending find
    std::optional<std::pair<bool, kmer_pair>> get_find_result(int id) {
        if (id >= 0 && id < pending_finds.size() && pending_finds[id].future.is_ready()) {
            return pending_finds[id].future.wait();
        }
        return std::nullopt;
    }

    // Process pending finds that are ready
    void process_pending_finds() {
        // Process progress to move communications forward
        upcxx::progress();
        
        // Check for batch flush opportunities
        for (int rank = 0; rank < upcxx::rank_n(); rank++) {
            if (find_buffers[rank].size() >= FIND_BATCH_SIZE / 2) {
                // Flush find buffer for this rank
                for (const auto& key : find_buffers[rank]) {
                    find_async(key);
                }
                find_buffers[rank].clear();
            }
        }
    }

    // Flush all insertion buffers asynchronously
    void flush_insertions_async() {
        for (int rank = 0; rank < upcxx::rank_n(); rank++) {
            if (!distributed_map->insertion_buffers[rank].empty()) {
                auto future = flush_buffer_for_rank(rank);
                pending_insertions.push_back(std::move(future));
            }
        }
        
        // Also flush find buffers
        for (int rank = 0; rank < upcxx::rank_n(); rank++) {
            if (!find_buffers[rank].empty()) {
                for (const auto& key : find_buffers[rank]) {
                    find_async(key);
                }
                find_buffers[rank].clear();
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
        bool result = insert_async(kmer);
        // Occasionally process pending operations to prevent backlog
        if (distributed_map->insertion_buffers[0].size() % 64 == 0) {
            process_pending_insertions();
        }
        return result;
    }

    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        // First check cache
        if (check_cache(key_kmer, val_kmer)) {
            return true;
        }
    
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

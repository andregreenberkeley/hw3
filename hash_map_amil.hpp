#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <memory>

struct HashMap {
    // Each rank stores a portion of the hash table
    std::vector<kmer_pair> data;
    std::vector<int> used;

    // Size of the local portion
    size_t my_size;
    
    // Global size of the hash table (across all ranks)
    size_t global_size;
    
    // Distributed object to access the hash table from any rank
    upcxx::dist_object<HashMap*> distributed_map;

    size_t size() const noexcept;
    size_t global_table_size() const noexcept;

    HashMap(size_t size);
    ~HashMap();

    // Insert a k-mer into the distributed hash map
    bool insert(const kmer_pair& kmer);
    
    // Find a k-mer in the distributed hash map
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
    
    // Distributed hash map functions
    uint64_t get_target_rank(const pkmer_t& kmer);
    uint64_t get_target_rank(uint64_t hash);
    uint64_t get_local_slot(uint64_t hash);
    
    // Remote operations via RPC
    bool remote_insert(const kmer_pair& kmer);
    bool remote_find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
};

HashMap::HashMap(size_t size) : distributed_map(this) {
    // Size per rank is total size divided by number of ranks
    my_size = (size + upcxx::rank_n() - 1) / upcxx::rank_n(); // Ceiling division
    global_size = my_size * upcxx::rank_n();
    
    data.resize(my_size);
    used.resize(my_size, 0);
}

HashMap::~HashMap() {
    // Nothing special needed for cleanup
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t target_rank = get_target_rank(hash);
    
    // If the target rank is this rank, do a local insert
    if (target_rank == upcxx::rank_me()) {
        uint64_t probe = 0;
        bool success = false;
        do {
            uint64_t slot = get_local_slot(hash + probe++);
            success = request_slot(slot);
            if (success) {
                write_slot(slot, kmer);
            }
        } while (!success && probe < my_size);
        return success;
    } else {
        // Otherwise, perform a remote insert using RPC
        return remote_insert(kmer);
    }
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t target_rank = get_target_rank(hash);
    
    // If the target rank is this rank, do a local find
    if (target_rank == upcxx::rank_me()) {
        uint64_t probe = 0;
        bool success = false;
        do {
            uint64_t slot = get_local_slot(hash + probe++);
            if (slot_used(slot)) {
                val_kmer = read_slot(slot);
                if (val_kmer.kmer == key_kmer) {
                    success = true;
                }
            } else {
                // If slot is not used, no need to continue
                break;
            }
        } while (!success && probe < my_size);
        return success;
    } else {
        // Otherwise, perform a remote find using RPC
        return remote_find(key_kmer, val_kmer);
    }
}

bool HashMap::slot_used(uint64_t slot) { 
    return used[slot] != 0; 
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    data[slot] = kmer; 
}

kmer_pair HashMap::read_slot(uint64_t slot) { 
    return data[slot]; 
}

bool HashMap::request_slot(uint64_t slot) {
    if (used[slot] != 0) {
        return false;
    } else {
        used[slot] = 1;
        return true;
    }
}

size_t HashMap::size() const noexcept { 
    return my_size; 
}

size_t HashMap::global_table_size() const noexcept {
    return global_size;
}

uint64_t HashMap::get_target_rank(const pkmer_t& kmer) {
    return get_target_rank(kmer.hash());
}

uint64_t HashMap::get_target_rank(uint64_t hash) {
    // Simple distribution: hash modulo number of ranks
    return hash % upcxx::rank_n();
}

uint64_t HashMap::get_local_slot(uint64_t hash) {
    // Get the slot in the local portion of the table
    // Using just the modulo ensures better distribution
    return hash % my_size;
}

bool HashMap::remote_insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t target_rank = get_target_rank(hash);
    
    // Use RPC to insert the k-mer on the target rank
    return upcxx::rpc(
        target_rank,
        [](upcxx::dist_object<HashMap*>& distributed_map, kmer_pair kmer) {
            // Get the local HashMap pointer on the target rank
            HashMap* local_map = *distributed_map;
            
            // Perform a local insert on the target rank
            uint64_t hash = kmer.hash();
            uint64_t probe = 0;
            bool success = false;
            do {
                uint64_t slot = local_map->get_local_slot(hash + probe++);
                success = local_map->request_slot(slot);
                if (success) {
                    local_map->write_slot(slot, kmer);
                }
            } while (!success && probe < local_map->size());
            
            return success;
        },
        distributed_map,
        kmer
    ).wait();
}

bool HashMap::remote_find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t target_rank = get_target_rank(hash);
    
    // Use RPC to find the k-mer on the target rank
    auto result = upcxx::rpc(
        target_rank,
        [](upcxx::dist_object<HashMap*>& distributed_map, pkmer_t key_kmer) {
            // Get the local HashMap pointer on the target rank
            HashMap* local_map = *distributed_map;
            
            // Perform a local find on the target rank
            uint64_t hash = key_kmer.hash();
            uint64_t probe = 0;
            bool success = false;
            kmer_pair found_kmer;
            
            do {
                uint64_t slot = local_map->get_local_slot(hash + probe++);
                if (local_map->slot_used(slot)) {
                    found_kmer = local_map->read_slot(slot);
                    if (found_kmer.kmer == key_kmer) {
                        success = true;
                    }
                } else {
                    // If slot is not used, no need to continue
                    break;
                }
            } while (!success && probe < local_map->size());
            
            // Return a pair containing success status and the k-mer (if found)
            return std::make_pair(success, success ? found_kmer : kmer_pair());
        },
        distributed_map,
        key_kmer
    ).wait();
    
    // Process the result
    bool success = result.first;
    if (success) {
        val_kmer = result.second;
    }
    
    return success;
}

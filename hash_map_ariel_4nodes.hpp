#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMapPart {
    upcxx::global_ptr<kmer_pair> data; // Global pointer to k-mer data
    upcxx::global_ptr<int> used;       // Global pointer to usage flags
    size_t size;                       // Local size on this rank
};

struct HashMap {
    upcxx::dist_object<HashMapPart> distributed_map; // Distributed object per rank
    size_t my_size;                                  // Local size
    size_t global_size;                              // Total size across all ranks
    upcxx::atomic_domain<int> ad_int;                // Atomic domain for synchronization

    HashMap(size_t size);
    ~HashMap();

    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    void flush_insertions();                         // New method to synchronize insertions

    size_t size() const noexcept { return global_size; }
};

HashMap::HashMap(size_t size)
    : distributed_map({nullptr, nullptr, 0}),
      ad_int({upcxx::atomic_op::compare_exchange, upcxx::atomic_op::load, upcxx::atomic_op::store}) {
    my_size = (size + upcxx::rank_n() - 1) / upcxx::rank_n(); // Ceiling division
    auto local_data = upcxx::new_array<kmer_pair>(my_size);
    auto local_used = upcxx::new_array<int>(my_size);
    int* local_used_ptr = local_used.local();
    for (size_t i = 0; i < my_size; i++) {
        local_used_ptr[i] = 0; // Initialize all slots as unused
    }
    *distributed_map = HashMapPart{local_data, local_used, my_size};
    global_size = upcxx::reduce_all(my_size, upcxx::op_fast_add).wait();
}

HashMap::~HashMap() {
    auto& part = *distributed_map;
    if (part.data) upcxx::delete_array(part.data);
    if (part.used) upcxx::delete_array(part.used);
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t target_rank = hash % upcxx::rank_n();
    auto remote_part = distributed_map.fetch(target_rank).wait();
    uint64_t start_slot = (hash / upcxx::rank_n()) % remote_part.size;

    for (uint64_t probe = 0; probe < remote_part.size; probe++) {
        uint64_t slot = (start_slot + probe) % remote_part.size;
        auto fut = ad_int.compare_exchange(remote_part.used + slot, 0, 1, std::memory_order_relaxed);
        int prev = fut.wait();
        if (prev == 0) { // Slot was empty, claim it
            upcxx::rput(kmer, remote_part.data + slot); // Non-blocking rput
            return true;
        }
        upcxx::progress(); // Process pending UPC++ operations
    }
    return false; // No empty slot found
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t target_rank = hash % upcxx::rank_n();
    auto remote_part = distributed_map.fetch(target_rank).wait();
    uint64_t start_slot = (hash / upcxx::rank_n()) % remote_part.size;

    for (uint64_t probe = 0; probe < remote_part.size; probe++) {
        uint64_t slot = (start_slot + probe) % remote_part.size;
        int used_val = upcxx::rget(remote_part.used + slot).wait();
        if (used_val == 0) return false; // Empty slot, k-mer not found
        val_kmer = upcxx::rget(remote_part.data + slot).wait();
        if (val_kmer.kmer == key_kmer) return true; // Found the k-mer
        upcxx::progress(); // Process pending operations
    }
    return false; // Searched all slots, not found
}

void HashMap::flush_insertions() {
    upcxx::barrier();          // Ensure all ranks have issued their rputs
    upcxx::rpc_ff(0, []() {}); // Dummy RPC to flush remote completions
    upcxx::barrier();          // Ensure all completions are processed
}

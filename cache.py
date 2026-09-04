# Delayed Momentum Cache (DeMoA principle)
import copy

class DelayedMomentumCache:
    def __init__(self, global_model, num_clients, aggregator_fn):
        self.num_clients = num_clients
        self.aggregator_fn = aggregator_fn
        initial_state = global_model.state_dict()
        self.cache = {cid: copy.deepcopy(initial_state) for cid in range(num_clients)}

    def step(self, sampled_ids, sampled_states):
        for cid, state in zip(sampled_ids, sampled_states):
            self.cache[cid] = copy.deepcopy(state)
            
        full_pool = [self.cache[cid] for cid in range(self.num_clients)]
        return self.aggregator_fn(full_pool)
import functools
import os
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

from demand_dists import (
    observe_demand_negbinom_PA,
    observe_demand_negbinom_PB,
    observe_demand_poisson,
    observe_demand_poisson_v2,
    true_lamb_poisson,
    true_lamb_poisson_v2,
    true_mean_demand_PA,
    true_mean_demand_PB,
)
from thompson_lp import TS_dynamic_oracle, TS_episodic_oracle
from utils import save_data

# Algorithm name list
AGENT_DIC = {
    "TSE*": TS_episodic_oracle,
    "TSD*": TS_dynamic_oracle,
}
# True demand distribution list
TRUE_DEMAND_DIST = {
    "poisson": observe_demand_poisson,
    "poisson_v2": observe_demand_poisson_v2,
    "negbinom_PA": observe_demand_negbinom_PA,
    "negbinom_PB": observe_demand_negbinom_PB,
}
# True demand list
TRUE_DEMAND = {
    "poisson": true_lamb_poisson,
    "poisson_v2": true_lamb_poisson_v2,
    "negbinom_PA": true_mean_demand_PA,
    "negbinom_PB": true_mean_demand_PB,
}


class exp:
    @classmethod
    def get_history(self, idx, random_state, param):
        # Set the true demand distribution
        observe_demand = TRUE_DEMAND_DIST[param["true_demand_dist"]]
        true_lamb = TRUE_DEMAND[param["true_demand_dist"]]
        # Form the true demand distribution as a function of p
        observe_demand_p = functools.partial(observe_demand, random_state=random_state)
        # Set an algorithm
        agent = AGENT_DIC[param["agent"]]

        print(f"run n{param['n0']} {param['agent']} {idx + 1}")

        # Get history
        cpu_start = time.process_time()
        history = agent.run(
            param["range_p"],
            param["p_shutoff"],
            param["range_t"],
            param["n0"],
            observe_demand_p,
            true_lamb,
            random_state,
        )
        cpu_time = time.process_time() - cpu_start

        return history, cpu_time

    @classmethod
    def repeat(self, param, processes):
        # generate different seeds for independent trials.
        random_states = [np.random.RandomState(seed) for seed in param["seeds"]]
        map_func = functools.partial(self.get_history, param=param)

        # Parallel computing
        with Pool(processes=processes) as pool:
            results = pool.starmap(map_func, enumerate(random_states))

        repeated_history, cpu_times = zip(*results)
        total_cpu_time = float(sum(cpu_times))

        return repeated_history, total_cpu_time

    @classmethod
    def element(self, param, processes):
        et = -time.perf_counter()
        repeated_history, total_cpu_time = self.repeat(param, processes)
        et += time.perf_counter()
        data = {
            "repeated_history": repeated_history,
            "elapsed_time": et,
            "total_cpu_time": total_cpu_time,
        }
        data.update(param)
        file_name = os.path.join(
            "data",
            "oracle",
            f"n{param['n0']}",
            f"{param['agent']}.pickle",
        )
        save_data(file_name, data)

    @classmethod
    def main(self):
        """
        Run experiments.
        """
        # set of candidate prices.
        range_p = np.arange(1, 10, 1)
        # shut-off price
        p_shutoff = 100000.0

        # Time horizon
        upper_t = 10.0
        # set time periods of each episode.
        range_t = np.arange(1, upper_t + 1, 1)

        # Initial inventory
        initial_inventories = [50, 30]

        # Number of independent trials.
        n_tri = 10000

        # Number of CPUs used for parallel computing.
        processes = 5

        # Common settings
        rng = np.random.RandomState(0)
        common_param = {
            "range_p": range_p,
            "p_shutoff": p_shutoff,
            "range_t": range_t,
            "true_demand_dist": "poisson",
            "seeds": [rng.randint(0, 2**32 - 1) for _ in range(n_tri)],
        }

        params = []
        params.extend(
            {
                **common_param,
                "n0": n0,
                "agent": agent,
            }
            for n0, agent in product(initial_inventories, AGENT_DIC)
        )

        for param in params:
            self.element(param=param, processes=processes)


if __name__ == "__main__":
    exp.main()

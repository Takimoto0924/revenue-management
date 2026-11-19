import functools
import os
import time
from itertools import product
from multiprocessing import Pool

import GPy
import numpy as np

from demand_dists import (
    observe_demand_negbinom_PA,
    observe_demand_negbinom_PB,
    observe_demand_poisson,
    observe_demand_poisson_v2,
)
from thompson_lp import TS_dynamic, TS_episodic
from utils import save_data

# Algorithm name list
AGENT_DIC = {
    "TSE": TS_episodic,
    "TSD": TS_dynamic,
}
# True demand distribution list
TRUE_DEMAND_DIST = {
    "poisson": observe_demand_poisson,
    "poisson_v2": observe_demand_poisson_v2,
    "negbinom_PA": observe_demand_negbinom_PA,
    "negbinom_PB": observe_demand_negbinom_PB,
}


class exp:
    @classmethod
    def get_history(self, idx, random_state, param):
        """1
        Run an experiment to obtain the resulting history.
        """
        # Set the true demand distribution
        observe_demand = TRUE_DEMAND_DIST[param["true_demand_dist"]]
        # Form the true demand distribution as a function of p
        observe_demand_p = functools.partial(observe_demand, random_state=random_state)
        # Set an algorithm
        agent = AGENT_DIC[param["agent"]]

        print(f"run {param['dist_type']} n{param['n0']} {param['agent']} {idx + 1}")

        # Get history
        cpu_start = time.process_time()
        history = agent.run(
            param["range_p"],
            param["p_shutoff"],
            param["range_t"],
            param["n0"],
            param["n_episodes"],
            observe_demand_p,
            param["dist_type"],
            param["dist_param"],
            random_state=random_state,
        )
        cpu_time = time.process_time() - cpu_start

        return history, cpu_time

    @classmethod
    def repeat(self, param, processes):
        """
        For parallel computation of experiments
        """
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
        """
        Get data from all independent trials and save it.
        """
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
            f"{param['dist_type']}",
            str(param["experiment_id"]),
            f"n{param['n0']}",
            f"{param['agent']}.pickle",
        )
        save_data(file_name, data)

    @classmethod
    def main(self):
        """
        Run experiments
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

        # Episode length
        n_episodes_gp = 200
        n_episodes_gamma = 5000
        # Number of independent trials.
        n_tri = 100

        # Number of CPUs used for parallel computing.
        processes = 5

        # Common settings
        rng = np.random.RandomState(0)
        common_param = {
            "experiment_id": 0,
            "range_p": range_p,
            "p_shutoff": p_shutoff,
            "range_t": range_t,
            "true_demand_dist": "poisson",
            "seeds": [rng.randint(0, 2**32 - 1) for _ in range(n_tri)],
        }
        # Hyper Parameters for a Gaussian process
        param_gp = {
            "dist_type": "gp",
            "dist_param": {
                "kernel_init": GPy.kern.RBF,
                "kernel_param": {"input_dim": 2, "ARD": True, "lengthscale": [3, 2.5]},
            },
            "n_episodes": n_episodes_gp,
        }
        # Hyper Parameters for the prior gamma
        param_gamma = {
            "dist_type": "gamma",
            "dist_param": {"init_alpha": 10.0, "init_beta": 1.0},
            "n_episodes": n_episodes_gamma,
        }

        params = []
        params.extend(
            {
                **common_param,
                **hyper_param,
                "n0": n0,
                "agent": agent,
            }
            for hyper_param, n0, agent in product(
                [param_gp, param_gamma],
                initial_inventories,
                AGENT_DIC,
            )
        )

        # Run the experiments for the setting given in every item of the list "params".
        for param in params:
            self.element(param=param, processes=processes)


if __name__ == "__main__":
    exp.main()

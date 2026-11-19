import cvxopt
import GPy
import numpy as np
from tqdm import tqdm


class TS_episodic:
    @classmethod
    def _init_lamb_dist(cls, range_p, range_t, dist_type, dist_param):
        if dist_type == "gp":
            return {"model": None}
        elif dist_type == "gamma":
            alpha = dist_param["init_alpha"] * np.ones(len(range_p) * len(range_t))
            beta = dist_param["init_beta"] * np.ones(len(range_p) * len(range_t))
            return {"alpha": alpha, "beta": beta}
        else:
            raise NotImplementedError

    @classmethod
    def _sample_demand_gp(
        cls, range_p, range_t, dist_param, lamb_dist, sample_slice, random_state
    ):
        """
        Sample mean demand from the Gaussian process.
        """
        upper_x0 = np.vstack([np.asarray([[t, p] for p in range_p]) for t in range_t])
        upper_x = upper_x0[sample_slice, :]

        # Initial samples
        if lamb_dist["model"]:
            mean, cov = lamb_dist["model"].predict(
                upper_x, full_cov=True, include_likelihood=False
            )
            mean = mean[:, 0]
        else:
            mean = np.zeros(upper_x.shape[0])
            cov = dist_param["kernel_init"](**dist_param["kernel_param"]).K(upper_x)
        # sampled latent functions using Gaussian process
        chol = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))

        f_sample = mean + chol @ random_state.normal(size=len(mean))
        lamb_sample = GPy.util.misc.safe_exp(f_sample)

        # return sampled mean demand
        return lamb_sample

    @classmethod
    def _sample_demand_gamma(cls, lamb_dist, sample_slice, random_state):
        """
        Sample mean demand from the posterior Gamma distributions.
        """
        # set gamma scale parameters
        alpha_part = lamb_dist["alpha"][sample_slice]
        beta_part = lamb_dist["beta"][sample_slice]
        # sample mean demand using the gamma distribution
        lamb_sample = random_state.gamma(shape=alpha_part, scale=1.0 / beta_part)

        return lamb_sample

    @classmethod
    def _sample_demand(
        cls,
        range_p,
        range_t,
        demand_model,
        sample_slice,
        random_state,
    ):
        """
        Return sampled mean demand from the posterior distribution.
        """
        dist_type = demand_model["dist_type"]
        dist_param = demand_model["dist_param"]
        lamb_dist = demand_model["lamb_dist"]

        # get mean demands through the Gaussian process
        if dist_type == "gp":
            return cls._sample_demand_gp(
                range_p, range_t, dist_param, lamb_dist, sample_slice, random_state
            )
        # get mean demands through the independent Gamma distribution
        elif dist_type == "gamma":
            return cls._sample_demand_gamma(lamb_dist, sample_slice, random_state)
        else:
            raise NotImplementedError

    @classmethod
    def _setup_c(cls, range_p, range_t, lamb_sample):
        """
        Set up the objective of the LP.
        """
        return np.hstack(
            [
                -range_p * lamb_sample[cnt * len(range_p) : (cnt + 1) * len(range_p)]
                for cnt in range(len(range_t))
            ]
        )

    @classmethod
    def _setup_g(cls, range_p, range_t, lamb_sample):
        """
        Set up the left-hand side of the constraints of the LP.
        """
        # inventory constraint
        block_lamb = np.hstack(
            [
                lamb_sample[cnt * len(range_p) : (cnt + 1) * len(range_p)]
                for cnt in range(len(range_t))
            ]
        )
        # constraint for the solution
        block_e = np.vstack(
            [
                np.hstack(
                    [
                        np.ones(len(range_p))
                        if cnt_c == cnt_r
                        else np.zeros(len(range_p))
                        for cnt_c in range(len(range_t))
                    ]
                )
                for cnt_r in range(len(range_t))
            ]
        )

        return np.vstack([block_lamb, block_e, -np.eye(len(range_p) * len(range_t))])

    @classmethod
    def _setup_h(cls, range_p, range_t, n_t):
        """
        Set up the right-hand side of the constraints of the LP.
        """
        return np.hstack(
            [[n_t], np.ones(len(range_t)), np.zeros(len(range_p) * len(range_t))]
        )

    @classmethod
    def _solve_lp(cls, c, upper_g, h):
        sol = cvxopt.solvers.lp(
            cvxopt.matrix(c),
            cvxopt.matrix(upper_g),
            cvxopt.matrix(h),
            options={"show_progress": False},
        )

        if sol["status"] != "optimal":
            print(sol["status"])

        return np.asarray(sol["x"]).ravel(), sol["status"]

    @classmethod
    def _calc_x(cls, range_p, range_t, n_t, lamb_sample):
        """
        Set up and solve the LP.
        """
        # set the objective function
        c = cls._setup_c(range_p, range_t, lamb_sample)
        # set the left-hand side of the constraints
        g = cls._setup_g(range_p, range_t, lamb_sample)
        # set the right-hand side of the constraints
        h = cls._setup_h(range_p, range_t, n_t)

        return cls._solve_lp(c, g, h)

    @classmethod
    def _calc_x_ext(cls, x):
        """
        Add the probability for the shut-off price
        """
        x_ext = np.append(x, 1.0 - np.sum(x))
        x_ext[x_ext < 0.0] = 0.0

        return x_ext / np.sum(x_ext)

    @classmethod
    def _sample_p(cls, range_p, p_shutoff, x, random_state):
        """
        Choose a price following the solution of the LP.
        """
        p_ext = np.append(range_p, p_shutoff)
        x_ext = cls._calc_x_ext(x)
        k = random_state.choice(len(p_ext), p=x_ext)

        return p_ext[k], k

    @classmethod
    def _update_n_and_revenue(cls, n_t, p, upper_d, revenue):
        """
        Consume inventory and obtain revenue observed demand based on observed demand.
        """
        # actual sales
        upper_d_prime = upper_d if n_t - upper_d >= 0.0 else n_t
        # inventory consumption
        n_t -= upper_d_prime
        # obtain revenue
        revenue += p * upper_d_prime

        return n_t, revenue

    @classmethod
    def _update_gp(cls, t, p, upper_d, dist_param, lamb_dist):
        """
        Update the Gaussian posterior distribution.
        """
        if isinstance(upper_d, np.ndarray):
            if upper_d.shape[1] != 1 or p.shape[1] != 1 or t.shape[1] != 1:
                raise RuntimeError
            x = np.hstack([t, p])
            y = upper_d
        else:
            x = np.asarray([[t, p]])
            y = np.asarray([[upper_d]])

        if lamb_dist["model"] is None:
            upper_x = x
            upper_y = y
        else:
            upper_x = np.vstack([lamb_dist["model"].X, x])
            upper_y = np.vstack([lamb_dist["model"].Y, y])

        # Set an inference method.
        inference_method = GPy.inference.latent_function_inference.Laplace()
        # Set a GPy instance
        lamb_dist["model"] = GPy.core.GP(
            X=upper_x,
            Y=upper_y,
            kernel=dist_param["kernel_init"](**dist_param["kernel_param"]),
            likelihood=GPy.likelihoods.Poisson(),
            inference_method=inference_method,
        )

        return lamb_dist

    @classmethod
    def _update_gp_once(cls, dist_param, lamb_dist, history):
        """
        Update the Gaussian process at the beginning of each episode.
        """
        # get the corresponding time period
        t = np.asarray([history[cnt]["t"] for cnt in range(len(history))]).reshape(
            -1, 1
        )
        # get the offered price
        p = np.asarray([history[cnt]["p"] for cnt in range(len(history))]).reshape(
            -1, 1
        )
        # get the observed demand
        upper_d = np.asarray(
            [history[cnt]["upper_d"] for cnt in range(len(history))]
        ).reshape(-1, 1)

        lamb_dist = cls._update_gp(t, p, upper_d, dist_param, lamb_dist)

        return lamb_dist

    @classmethod
    def _update_gamma(cls, cnt_t, k, upper_d, range_p, lamb_dist):
        """
        Update the gamma posterior distribution.
        """
        if not isinstance(upper_d, (list, np.ndarray)):
            upper_d = [upper_d]
            k = [k]
            cnt_t = [cnt_t]

        for cnt_d in range(len(upper_d)):
            if k[cnt_d] >= len(range_p):
                continue
            idx = cnt_t[cnt_d] * len(range_p) + k[cnt_d]
            lamb_dist["alpha"][idx] += upper_d[cnt_d]
            lamb_dist["beta"][idx] += 1.0

        return lamb_dist

    @classmethod
    def _update_gamma_once(cls, range_p, lamb_dist, history):
        """
        Update the gamma distribution at the beginning of each episode.
        """
        cnt_t = np.asarray([history[cnt]["cnt_t"] for cnt in range(len(history))])
        k = np.asarray([history[cnt]["k"] for cnt in range(len(history))])
        upper_d = np.asarray([history[cnt]["upper_d"] for cnt in range(len(history))])

        lamb_dist = cls._update_gamma(cnt_t, k, upper_d, range_p, lamb_dist)

        return lamb_dist

    @classmethod
    def _update_dist_once(cls, range_p, demand_model, history, random_state):
        """
        Update the posterior distribution at the beginning of each episode.
        """
        dist_type = demand_model["dist_type"]
        dist_param = demand_model["dist_param"]
        lamb_dist = demand_model["lamb_dist"]

        if dist_type == "gp":
            lamb_dist = cls._update_gp_once(dist_param, lamb_dist, history)
        elif dist_type == "gamma":
            lamb_dist = cls._update_gamma_once(range_p, lamb_dist, history)
        else:
            raise NotImplementedError

        demand_model["lamb_dist"] = lamb_dist

    @classmethod
    def episode(
        cls,
        range_p,
        p_shutoff,
        range_t,
        n0,
        observe_demand,
        demand_model,
        random_state,
    ):
        """
        Run an episode of an episodic revenue management problem.
        """
        n_t = n0
        revenue = 0.0
        episode_history = []

        for cnt_t, t in enumerate(range_t):
            if cnt_t == 0:
                # Sample demand
                lamb_sample = cls._sample_demand(
                    range_p,
                    range_t,
                    demand_model,
                    slice(None),
                    random_state,
                )

                # Optimize prices given sampled demand
                x, x_status = cls._calc_x(range_p, range_t, n_t, lamb_sample)

            # Restrict the solution for a suitable form at this period
            period_slice = slice(cnt_t * len(range_p), (cnt_t + 1) * len(range_p))
            lamb_sample_t = lamb_sample[period_slice]
            x_t = x[period_slice]

            # Offer price
            p, k = cls._sample_p(range_p, p_shutoff, x_t, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue = cls._update_n_and_revenue(n_t, p, upper_d, revenue)

            # Add the obtained data to the history
            episode_history.append(
                {
                    "t": t,
                    "cnt_t": cnt_t,
                    "lamb_sample": lamb_sample_t,
                    "x": x_t,
                    "x_status": x_status,
                    "p": p,
                    "k": k,
                    "upper_d": upper_d,
                    "inventory_begin": inventory_begin,
                    "inventory_end": n_t,
                    "revenue": revenue,
                }
            )

            if n_t == 0.0:
                break

        if isinstance(demand_model, dict):
            # Update the posterior distribution
            cls._update_dist_once(range_p, demand_model, episode_history, random_state)

        return episode_history

    @classmethod
    def run(
        cls,
        range_p,
        p_shutoff,
        range_t,
        n0,
        n_episodes,
        observe_demand,
        dist_type,
        dist_param,
        random_state,
    ):
        """
        Run an episodic revenue management problem.
        """
        demand_model = {
            "dist_type": dist_type,
            "dist_param": dist_param,
            "lamb_dist": cls._init_lamb_dist(range_p, range_t, dist_type, dist_param),
        }

        history = []
        for _ in tqdm(range(n_episodes), leave=False):
            episode_history = cls.episode(
                range_p,
                p_shutoff,
                range_t,
                n0,
                observe_demand,
                demand_model,
                random_state,
            )
            history.append(episode_history)

        return history


class TS_dynamic(TS_episodic):
    @classmethod
    def _update_dist(
        cls,
        t,
        cnt_t,
        p,
        k,
        upper_d,
        range_p,
        demand_model,
        random_state,
    ):
        dist_type = demand_model["dist_type"]
        dist_param = demand_model["dist_param"]
        lamb_dist = demand_model["lamb_dist"]

        if dist_type in ("gp", "monotone_gp"):
            lamb_dist = cls._update_gp(t, p, upper_d, dist_param, lamb_dist)
        elif dist_type in ("gamma", "monotone_gamma"):
            lamb_dist = cls._update_gamma(cnt_t, k, upper_d, range_p, lamb_dist)
        elif dist_type == "gamma_diff":
            lamb_dist = cls._update_gamma_diff(
                cnt_t, k, upper_d, lamb_dist, random_state
            )
        else:
            raise NotImplementedError

        demand_model["lamb_dist"] = lamb_dist

    @classmethod
    def episode(
        cls,
        range_p,
        p_shutoff,
        range_t,
        n0,
        observe_demand,
        demand_model,
        random_state,
    ):
        n_t = n0
        revenue = 0.0
        episode_history = []

        for cnt_t, t in enumerate(range_t):
            # Sample demand
            lamb_sample = cls._sample_demand(
                range_p,
                range_t,
                demand_model,
                slice(cnt_t * len(range_p), None),
                random_state,
            )

            # Optimize prices given sampled demand
            x, x_status = cls._calc_x(range_p, range_t[cnt_t:], n_t, lamb_sample)

            # Restrict the solution for a suitable form at this period
            period_slice = slice(0, len(range_p))
            lamb_sample_t = lamb_sample[period_slice]
            x_t = x[period_slice]

            # Offer price
            p, k = cls._sample_p(range_p, p_shutoff, x_t, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue = cls._update_n_and_revenue(n_t, p, upper_d, revenue)

            if isinstance(demand_model, dict):
                # Update the posterior distribution.
                cls._update_dist(
                    t,
                    cnt_t,
                    p,
                    k,
                    upper_d,
                    range_p,
                    demand_model,
                    random_state,
                )

            # Add the obtained data to the history
            episode_history.append(
                {
                    "t": t,
                    "cnt_t": cnt_t,
                    "lamb_sample": lamb_sample_t,
                    "x": x_t,
                    "x_status": x_status,
                    "p": p,
                    "k": k,
                    "upper_d": upper_d,
                    "inventory_begin": inventory_begin,
                    "inventory_end": n_t,
                    "revenue": revenue,
                }
            )

            if n_t == 0.0:
                break

        return episode_history


class TS_oracle:
    @classmethod
    def _sample_demand(cls, range_p, range_t, true_lamb, sample_slice, random_state):
        """
        Get true mean demand.
        """
        upper_x0 = np.vstack([np.asarray([[t, p] for p in range_p]) for t in range_t])
        upper_x = upper_x0[sample_slice, :]
        upper_t = upper_x[-1][0]

        return np.asarray([true_lamb(x[1], x[0], upper_t) for x in upper_x])

    @classmethod
    def episode(
        cls,
        range_p,
        p_shutoff,
        range_t,
        n0,
        observe_demand,
        true_lamb,
        random_state,
    ):
        episode_history = super().episode(
            range_p,
            p_shutoff,
            range_t,
            n0,
            observe_demand,
            true_lamb,
            random_state,
        )

        return episode_history

    @classmethod
    def run(
        cls,
        range_p,
        p_shutoff,
        range_t,
        n0,
        observe_demand,
        true_lamb,
        random_state,
    ):
        history = cls.episode(
            range_p,
            p_shutoff,
            range_t,
            n0,
            observe_demand,
            true_lamb,
            random_state,
        )

        return [history]


class TS_episodic_oracle(TS_oracle, TS_episodic):
    pass


class TS_dynamic_oracle(TS_oracle, TS_dynamic):
    pass

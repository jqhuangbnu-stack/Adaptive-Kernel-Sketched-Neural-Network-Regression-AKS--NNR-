"""Data generator for the ten article models A--J.

A--E have d*=2; F--J have d*=3.  The public API is compatible with the
original project: generate_X, generate_Y, generate_dataset, and
generate_single_config.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


class DataGenerator:
    D2_MODELS = {"A", "B", "C", "D", "E"}
    D3_MODELS = {"F", "G", "H", "I", "J"}
    ARTICLE_MODELS = D2_MODELS | D3_MODELS

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def true_dimension(model):
        model = str(model).upper()
        if model in DataGenerator.D2_MODELS:
            return 2
        if model in DataGenerator.D3_MODELS:
            return 3
        raise ValueError(f"Unknown article model: {model}. Use A--J.")

    @staticmethod
    def setting_family(model):
        return f"article_d{DataGenerator.true_dimension(model)}"

    def generate_X(self, p, n, scenario="I", cov_structure=None):
        scenario = str(scenario).upper()
        if scenario == "I":
            return np.random.normal(0.0, np.sqrt(0.5), size=(n, p))
        if scenario == "II":
            return -1.0 + np.random.poisson(lam=1.0, size=(n, p))
        if scenario == "III":
            cov = (0.6 * np.eye(p) + 0.4 * np.ones((p, p)))
            if cov_structure is not None:
                cov = cov_structure
            z = np.random.multivariate_normal(np.zeros(p), cov, size=n)
            v = np.random.chisquare(4, size=n).reshape(-1, 1)
            return z / np.sqrt(v / 4.0)

        # Compatibility aliases used by earlier scripts.
        if scenario == "GA":
            return np.random.normal(0.0, 1.0, size=(n, p))
        if scenario == "GB":
            signs = np.random.binomial(1, 0.5, size=(n, 1))
            means = np.where(signs == 1, 1.0, -1.0)
            return np.random.normal(means, 1.0, size=(n, p))
        if scenario == "GC":
            cov = 0.6 * np.eye(p) + 0.4 * np.ones((p, p))
            return np.random.multivariate_normal(np.zeros(p), cov, size=n)
        raise ValueError(f"Unknown scenario: {scenario}. Use I/II/III or GA/GB/GC.")

    @staticmethod
    def _z(X, j):
        return X[:, j] - X[:, j + 1]

    @staticmethod
    def _eps(n, q, noise_std, random_noise):
        if not random_noise:
            return np.zeros((n, q))
        return np.random.normal(0.0, noise_std, size=(n, q))

    def _d2(self, X, model, noise_std, random_noise):
        n, p = X.shape
        if model in {"A", "B"} and p < 4:
            raise ValueError(f"Model {model} requires p >= 4.")
        if model in {"C", "D", "E"} and p < 8:
            raise ValueError(f"Model {model} requires p >= 8.")

        if model == "A":
            eta = np.column_stack([
                4.0 * self._z(X, 0) / np.sqrt(1.0 + self._z(X, 0) ** 2),
                4.0 * self._z(X, 2) / np.sqrt(1.0 + self._z(X, 2) ** 2),
            ])
            return eta + self._eps(n, 2, noise_std, random_noise), eta

        if model == "B":
            e1, e2 = 2.0 * np.sin(self._z(X, 0)), 2.0 * np.sin(self._z(X, 2))
            eta = np.column_stack([e1, e2])
            y = np.column_stack([e1, e2, e1 + e2, e1 - e2])
            return y + self._eps(n, 4, noise_std, random_noise), eta

        if model == "C":
            z1, w1 = self._z(X, 0), self._z(X, 2)
            z2, w2 = self._z(X, 4), self._z(X, 6)
            eta = np.column_stack([
                2.0 * z1 / (1.0 + np.exp(np.clip(w1, -40, 40))),
                2.0 * z2 / (1.0 + np.exp(np.clip(w2, -40, 40))),
            ])
            return eta + self._eps(n, 2, noise_std, random_noise), eta

        if model == "D":
            z1, w1 = self._z(X, 0), self._z(X, 2)
            z2, w2 = self._z(X, 4), self._z(X, 6)
            eta = np.column_stack([
                3.0 * np.tanh(z1) * (1.0 + 0.5 * np.tanh(w1)),
                3.0 * z2 * (1.0 + 0.5 * np.sin(w2)) / np.sqrt(1.0 + z2 ** 2),
            ])
            return eta + self._eps(n, 2, noise_std, random_noise), eta

        if model == "E":
            z1, z2 = self._z(X, 0), self._z(X, 2)
            z3, z4 = self._z(X, 4), self._z(X, 6)
            e1 = 3.0 * np.tanh(z1) + 0.8 * np.tanh(z2)
            e2 = 2.0 * np.sin(z3) + 0.8 * z4 / np.sqrt(1.0 + z4 ** 2)
            eta = np.column_stack([e1, e2])
            y = np.column_stack([
                e1 + 0.15 * np.sin(e1 + e2),
                e2 + 0.15 * np.tanh(e1 + 2.0 * e2),
            ])
            return y + self._eps(n, 2, noise_std, random_noise), eta

        raise ValueError(f"Unknown d=2 article model: {model}")

    def _d3(self, X, model, noise_std, random_noise):
        n, p = X.shape
        if p < 6:
            raise ValueError(f"Model {model} requires p >= 6.")
        if model in {"H", "J"} and p < 12:
            raise ValueError(f"Model {model} requires p >= 12.")

        if model == "F":
            eta = np.column_stack([
                4.0 * self._z(X, j) / np.sqrt(1.0 + self._z(X, j) ** 2)
                for j in (0, 2, 4)
            ])
            return eta + self._eps(n, 3, noise_std, random_noise), eta

        if model == "G":
            z1, z2, z3 = self._z(X, 0), self._z(X, 2), self._z(X, 4)
            e1 = 3.0 * z1
            e2 = 2.0 * z2 + 2.0 * np.sin(z2)
            e3 = 1.5 * z3 + 3.0 * z3 / np.sqrt(1.0 + z3 ** 2)
            eta = np.column_stack([e1, e2, e3])
            y = np.column_stack([e1, e2, e3, 0.4 * (e1 + e2 + e3)])
            return y + self._eps(n, 4, noise_std, random_noise), eta

        if model == "H":
            z1, z2, z3 = self._z(X, 0), self._z(X, 2), self._z(X, 4)
            w1, w2, w3 = self._z(X, 2), self._z(X, 6), self._z(X, 10)
            eta = np.column_stack([
                2.0 * z1 / (1.0 + np.exp(np.clip(w1, -40, 40))),
                2.0 * z2 / (1.0 + np.exp(np.clip(w2, -40, 40))),
                2.0 * z3 / (1.0 + np.exp(np.clip(w3, -40, 40))),
            ])
            return eta + self._eps(n, 3, noise_std, random_noise), eta

        if model == "I":
            z1, z2, z3 = self._z(X, 0), self._z(X, 2), self._z(X, 4)
            e1, e2, e3 = 2.0 * np.sin(z1), 2.0 * np.cos(z2), 1.5 * np.log(1.0 + z3 ** 2)
            eta = np.column_stack([e1, e2, e3])
            y = np.column_stack([e1, e2, e3, e1 + e2, e1 - e3, e2 + e3])
            return y + self._eps(n, 6, noise_std, random_noise), eta

        if model == "J":
            z1, z2, z3 = self._z(X, 0), self._z(X, 2), self._z(X, 4)
            z4, z5, z6 = self._z(X, 6), self._z(X, 8), self._z(X, 10)
            e1 = 3.0 * np.tanh(z1) + 0.8 * np.tanh(z2)
            e2 = 2.0 * np.sin(z3) + 0.8 * z4 / np.sqrt(1.0 + z4 ** 2)
            e3 = 3.0 * z5 / np.sqrt(1.0 + z5 ** 2) + 0.8 * np.tanh(z6)
            eta = np.column_stack([e1, e2, e3])
            h1 = e1 + 0.15 * np.sin(e1 + e2)
            h2 = e2 + 0.15 * np.tanh(e1 + 2.0 * e2)
            h3 = e3 + 0.15 * (e2 + e3) / np.sqrt(1.0 + (e2 + e3) ** 2)
            y = np.column_stack([h1, h2, h3, 0.4 * h1 + 0.4 * h2 - 0.4 * h3])
            return y + self._eps(n, 4, noise_std, random_noise), eta

        raise ValueError(f"Unknown d=3 article model: {model}")

    def generate_Y(self, X, model="A", noise_std=0.5, random_noise=True):
        model = str(model).upper()
        if model not in self.ARTICLE_MODELS:
            raise ValueError(f"Unknown article model: {model}. Use A, B, ..., J.")
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must be a two-dimensional array with p >= 2.")
        return (self._d2 if model in self.D2_MODELS else self._d3)(
            X, model, noise_std, random_noise
        )

    def generate_dataset(self, p=50, n=1000, m=1000, model="A", scenario="I",
                         test_same_dist=True, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        model, scenario = str(model).upper(), str(scenario).upper()
        X_train = self.generate_X(p, n, scenario)
        Y_train, eta_train = self.generate_Y(X_train, model)
        test_scenario = scenario if test_same_dist else scenario
        X_test = self.generate_X(p, m, test_scenario)
        Y_test, _ = self.generate_Y(X_test, model, random_noise=True)
        Y_true, eta_test = self.generate_Y(X_test, model, random_noise=False)
        return {
            "X_train": X_train, "Y_train": Y_train, "eta_train": eta_train,
            "X_test": X_test, "Y_test": Y_test, "Y_test_true": Y_true,
            "eta_test": eta_test, "model": model, "scenario": scenario,
            "setting_family": self.setting_family(model),
            "true_dim": self.true_dimension(model),
            "response_dim": 1 if np.asarray(Y_train).ndim == 1 else Y_train.shape[1],
            "p": p, "n": n, "m": m, "random_state": random_state,
        }


def generate_single_config(p=50, n=1000, m=1000, model="A", scenario="I",
                           random_state=None):
    return DataGenerator().generate_dataset(
        p=p, n=n, m=m, model=model, scenario=scenario,
        random_state=random_state
    )

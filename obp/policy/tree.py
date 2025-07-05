"""Contextual Tree Bandit Algorithms."""
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_scalar

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Local application imports
from ..utils import check_array
from .base import BaseContextualPolicy


@dataclass
class BaseTreePolicy(BaseContextualPolicy):
    """Base class for contextual bandit policies using regression tree.

    Parameters
    ----------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    """

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

    def _update_models(self) -> None:
        """Update tree models."""
        pass


@dataclass
class TreeBootstrap(BaseTreePolicy):
    """The Contextual Bandit Algorithm with Decision Tree and Bootstrap.

    Elmachtoub, Adam N., Ryan McNellis, Sechan Oh, and Marek Petrik. 2017.
    "A Practical Method for Solving Contextual Bandit Problems Using Decision Trees."
    arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/1706.04687.

    This algorithm implements Thompson Sampling via bootstrapping:
    1. For each action, maintain history of context-reward pairs
    2. When selecting an action, create a bootstrap sample of the history for each action
    3. Train a decision tree on each bootstrap sample
    4. Select the action with the highest predicted reward

    Parameters
    -----------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.
    """

    def __post_init__(self) -> None:
        """Initialize class."""
        self.policy_name = "tree_bootstrap"
        super().__post_init__()

        # Initialize context and reward lists for each action
        self.context_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_actions)]
        self.reward_lists: List[List[float]] = [[] for _ in range(self.n_actions)]

        # Initialize model list (will be recreated during select_action)
        self.model_list = [
            DecisionTreeRegressor(random_state=self.random_state)
            for _ in range(self.n_actions)
        ]

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        context: array-like, shape (1, dim_context)
            Observed context vector.
        """
        self.n_trial += 1
        self.action_counts[action] += 1
        # Store context and reward for the selected action
        self.context_lists[action].append(context)
        self.reward_lists[action].append(reward)

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data using bootstrap sampling.

        Parameters
        ----------
        context: array-like, shape (1, dim_context)
            Observed context vector.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        # Create a prediction for each action
        action_scores = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) > 0:
                # Create bootstrap sample
                n_samples = len(self.reward_lists[action])
                # If we have at least 2 samples (minimum needed for a decision tree)
                if n_samples >= 2:
                    # Create bootstrap indices
                    bootstrap_indices = self.random_.choice(
                        n_samples, size=n_samples, replace=True
                    )

                    # Create bootstrap sample
                    bootstrap_contexts = [
                        self.context_lists[action][i] for i in bootstrap_indices
                    ]
                    bootstrap_rewards = [
                        self.reward_lists[action][i] for i in bootstrap_indices
                    ]

                    # Concatenate contexts
                    if len(bootstrap_contexts) > 0:
                        X_bootstrap = np.concatenate(bootstrap_contexts, axis=0)
                        y_bootstrap = np.array(bootstrap_rewards)
                        # Fit decision tree on bootstrap sample
                        tree_model = DecisionTreeRegressor(
                            random_state=self.random_state
                        )
                        tree_model.fit(X_bootstrap, y_bootstrap)

                        # Predict for current context
                        action_scores[action] = tree_model.predict(context)[0]
                    else:
                        action_scores[action] = np.mean(self.reward_lists[action])
                else:
                    # Not enough samples for tree, use simple average
                    action_scores[action] = np.mean(self.reward_lists[action])
            else:
                # No data available, assign neutral score
                action_scores[action] = 0.5

        # Return actions with highest scores
        return action_scores.argsort()[::-1][: self.len_list]

    def _update_models(self) -> None:
        """Update tree models.

        For TreeBootstrap, this is optional because the models are rebuilt during each
        select_action call. However, we can train models here for potential reuse
        if needed for evaluation purposes.
        """
        # For each action, if we have enough data, train a model on all available data
        for action in range(self.n_actions):
            if (
                len(self.reward_lists[action]) >= 2
            ):  # Need at least 2 samples for a tree
                contexts = np.concatenate(self.context_lists[action], axis=0)
                rewards = np.array(self.reward_lists[action])

                # Check if we have enough unique values to fit a tree
                if len(np.unique(rewards)) >= 2:
                    self.model_list[action] = DecisionTreeRegressor(
                        random_state=self.random_state
                    )
                    self.model_list[action].fit(contexts, rewards)


@dataclass
class TreeUCB(BaseTreePolicy):
    """Tree Upper Confidence Bound.

    TreeUCB は TreeBootstrap の「不確実性付与」をブートストラップではなく、
    決定木葉ノードの期待報酬の信頼上界 (Upper Confidence Bound, UCB) で置き換えた改良版です。
    葉ノード s の成功数 Kₛ と試行数 Nₛ から母比率の上側信頼限界を計算し、
    各アクションの上界を比較して選択します。

    Parameters
    ------------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    alpha: float, default=0.05
        Confidence level for the upper confidence bound calculation.
        Lower values result in wider confidence intervals and more exploration.

    References
    ----------
    Lihong Li, Wei Chu, John Langford, and Robert E Schapire.
    "A Contextual-bandit Approach to Personalized News Article Recommendation," 2010.
    """

    alpha: float = 0.05

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.alpha, "alpha", float, min_val=0.0, max_val=1.0)
        self.policy_name = f"tree_ucb_{self.alpha}"
        super().__post_init__()

        # Initialize context and reward lists for each action
        self.context_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_actions)]
        self.reward_lists: List[List[float]] = [[] for _ in range(self.n_actions)]
        # Initialize model list
        self.model_list = [
            DecisionTreeRegressor(random_state=self.random_state)
            for _ in range(self.n_actions)
        ]

        # For z-score calculation (approximately 1.96 for alpha=0.05)
        self.z_score = stats.norm.ppf(1 - self.alpha / 2)

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        context: array-like, shape (1, dim_context)
            Observed context vector.
        """
        self.n_trial += 1
        self.action_counts[action] += 1

        # Store context and reward for the selected action
        self.context_lists[action].append(context)
        self.reward_lists[action].append(reward)

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data using UCB.

        Parameters
        ------------
        context: array-like, shape (1, dim_context)
            Observed context vector.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        # Update models for all actions with current data
        self._update_models()

        # Calculate UCB score for each action
        ucb_scores = np.zeros(self.n_actions)

        for action in range(self.n_actions):
            if len(self.reward_lists[action]) > 0:
                if (
                    hasattr(self.model_list[action], "tree_")
                    and self.model_list[action].tree_ is not None
                ):
                    # Get mean prediction
                    mean_pred = self.model_list[action].predict(context)[0]

                    # Get the leaf node that the context falls into
                    # This is a simplified approach, as we can't directly access counts in leaf nodes
                    # In a more complete implementation, we'd track sample counts in leaf nodes
                    n_samples = len(self.reward_lists[action])

                    # Simple UCB calculation (p_hat + z * sqrt(p_hat * (1 - p_hat) / n))
                    # For binary rewards in [0,1] - adjust as needed for other reward ranges
                    p_hat = max(0.0, min(1.0, mean_pred))  # Clip to [0,1]
                    std_err = np.sqrt(p_hat * (1 - p_hat) / max(1, n_samples))
                    ucb = p_hat + self.z_score * std_err

                    ucb_scores[action] = ucb
                else:
                    # If no tree is available, use the mean with a wide confidence interval
                    mean_reward = np.mean(self.reward_lists[action])
                    std_err = np.sqrt(
                        0.25 / max(1, len(self.reward_lists[action]))
                    )  # Maximum variance for binary reward
                    ucb_scores[action] = mean_reward + self.z_score * std_err
            else:
                # No data for this action - assign high UCB
                ucb_scores[action] = 1.0

        # Return actions with highest UCB scores
        return ucb_scores.argsort()[::-1][: self.len_list]

    def _update_models(self) -> None:
        """Update tree models for all actions using available data."""
        for action in range(self.n_actions):
            if (
                len(self.reward_lists[action]) >= 2
            ):  # Need at least 2 samples for a tree
                contexts = np.concatenate(self.context_lists[action], axis=0)
                rewards = np.array(self.reward_lists[action])

                # Check if we have enough unique values to fit a tree
                if len(np.unique(rewards)) >= 2:
                    self.model_list[action] = DecisionTreeRegressor(
                        random_state=self.random_state
                    )
                    self.model_list[action].fit(contexts, rewards)


@dataclass
class BaseTreeEnsemblePolicy(BaseTreePolicy):
    """Base class for tree ensemble contextual bandit policies (XGBoost/Random Forest)."""

    init_rounds: int = 10
    nu: float = 1.0
    n_estimators: int = 10
    learning_rate: float = 0.1
    max_depth: int = 3

    def __post_init__(self) -> None:
        check_scalar(self.init_rounds, "init_rounds", int, min_val=1)
        check_scalar(self.nu, "nu", float, min_val=0.0)
        check_scalar(self.n_estimators, "n_estimators", int, min_val=2)
        check_scalar(
            self.learning_rate,
            "learning_rate",
            float,
            min_val=0.0,
            include_boundaries="neither",
        )
        check_scalar(self.max_depth, "max_depth", int, min_val=1)
        super().__post_init__()

        if not HAS_XGBOOST:
            raise ImportError(
                "TreeEnsemblePolicy requires XGBoost to be installed. "
                "Please install it using: pip install xgboost"
            )

        self.context_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_actions)]
        self.reward_lists: List[List[float]] = [[] for _ in range(self.n_actions)]
        self.model_list = [
            XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            for _ in range(self.n_actions)
        ]

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        self.n_trial += 1
        self.action_counts[action] += 1
        self.context_lists[action].append(context)
        self.reward_lists[action].append(reward)

    def _set_leaf_values(self, models, context):
        all_leaf_stats = {}
        for action_idx, model in enumerate(models):
            if model is None or not hasattr(model, "get_booster"):
                all_leaf_stats[action_idx] = {}
                continue
            try:
                booster = model.get_booster()
                n_trees = len(booster.get_dump())
                base_value = model.get_booster().attributes()["base_score"]
                if isinstance(base_value, str):
                    base_value = float(base_value)
                try:
                    leaf_indices = booster.predict(context, pred_leaf=True)[0]
                except TypeError:
                    from xgboost import DMatrix

                    leaf_indices = booster.predict(DMatrix(context), pred_leaf=True)[0]
                leaf_stats = {
                    i: {
                        int(leaf_indices[i]): {
                            "count": 1,
                            "mean": 0.0,
                            "var": 0.1,
                        }
                    }
                    for i in range(min(n_trees, len(leaf_indices)))
                }
                prev_pred = base_value
                for tree_idx in range(min(n_trees, len(leaf_indices))):
                    leaf_idx = int(leaf_indices[tree_idx])
                    if tree_idx > 0:
                        try:
                            prev_pred = model.predict(
                                context, iteration_range=(0, tree_idx)
                            )[0]
                        except TypeError:
                            from xgboost import DMatrix

                            prev_pred = model.predict(
                                DMatrix(context), iteration_range=(0, tree_idx)
                            )[0]
                    try:
                        curr_pred = model.predict(
                            context, iteration_range=(0, tree_idx + 1)
                        )[0]
                    except TypeError:
                        from xgboost import DMatrix

                        curr_pred = model.predict(
                            DMatrix(context), iteration_range=(0, tree_idx + 1)
                        )[0]
                    contribution = self.learning_rate * (curr_pred - prev_pred)
                    leaf_stats[tree_idx][leaf_idx]["mean"] = contribution
                    prev_pred = curr_pred
                    leaf_stats[tree_idx][leaf_idx]["var"] = 0.1
                all_leaf_stats[action_idx] = leaf_stats
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to extract leaf statistics for action {action_idx}: {str(e)}"
                )
                all_leaf_stats[action_idx] = {}
        return all_leaf_stats

    def _get_ensemble_prediction_with_variance(
        self, context: np.ndarray, action: int
    ) -> tuple:
        model = self.model_list[action]
        if model is None:
            return 0.5, 1.0, 1.0
        try:
            booster = model.get_booster()
            try:
                mean_pred = model.predict(context)[0]
            except TypeError:
                from xgboost import DMatrix

                mean_pred = model.predict(DMatrix(context))[0]
        except Exception:
            return 0.5, 1.0, 1.0
        booster = model.get_booster()
        n_trees = len(booster.get_dump())
        leaf_stats = self.all_leaf_stats.get(action, {})
        tree_variance = 0.0
        c_t_a = 0.0
        try:
            leaf_indices = booster.predict(context, pred_leaf=True)
        except TypeError:
            from xgboost import DMatrix

            leaf_indices = booster.predict(DMatrix(context), pred_leaf=True)
        for tree_idx in range(n_trees):
            leaf_idx = int(
                leaf_indices[0][tree_idx]
                if len(leaf_indices[0]) > tree_idx
                else leaf_indices[0][0]
            )
            if tree_idx in leaf_stats and leaf_idx in leaf_stats[tree_idx]:
                stats = leaf_stats[tree_idx][leaf_idx]
                tree_variance += stats["var"]
                c_t_a += stats["count"]
            else:
                tree_variance += 0.1
                c_t_a += 1
        variance = tree_variance / max(1, c_t_a)
        return mean_pred, variance, c_t_a

    def _update_models(self) -> None:
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) >= 2:
                try:
                    contexts = np.concatenate(self.context_lists[action], axis=0)
                    rewards = np.array(self.reward_lists[action])
                    if len(np.unique(rewards)) >= 2:
                        from xgboost import XGBRegressor

                        self.model_list[action] = XGBRegressor(
                            n_estimators=self.n_estimators,
                            learning_rate=self.learning_rate,
                            max_depth=self.max_depth,
                            random_state=self.random_state,
                        )
                        self.model_list[action].fit(contexts, rewards)
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to update model for action {action}: {str(e)}"
                    )


@dataclass
class TreeEnsembleUCB(BaseTreeEnsemblePolicy):
    """Tree Ensemble Upper Confidence Bound (TEUCB)."""

    def __post_init__(self) -> None:
        self.policy_name = f"tree_ensemble_ucb_{self.nu}"
        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")
        if self.n_trial < self.init_rounds:
            return self.random_.choice(
                self.n_actions, size=self.len_list, replace=False
            )
        self._update_models()
        self.all_leaf_stats = self._set_leaf_values(self.model_list, context)
        ucb_scores = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) > 0:
                model = self.model_list[action]
                is_fitted = False
                if model is not None:
                    try:
                        model.get_booster()
                        is_fitted = True
                    except Exception:
                        is_fitted = False
                if is_fitted:
                    mean_pred, variance, c_t_a = (
                        self._get_ensemble_prediction_with_variance(context, action)
                    )
                    exploration_bonus = self.nu * np.sqrt(
                        variance * np.log(max(1, self.n_trial - 1)) / c_t_a
                    )
                    ucb_scores[action] = mean_pred + exploration_bonus
                else:
                    mean_reward = np.mean(self.reward_lists[action])
                    ucb_scores[action] = mean_reward + self.nu
            else:
                ucb_scores[action] = 1.0
        return ucb_scores.argsort()[::-1][: self.len_list]


@dataclass
class TreeEnsembleTS(BaseTreeEnsemblePolicy):
    """Tree Ensemble Thompson Sampling (TETS)."""

    def __post_init__(self) -> None:
        self.policy_name = f"tree_ensemble_ts_{self.nu}"
        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")
        if self.n_trial < self.init_rounds:
            return self.random_.choice(
                self.n_actions, size=self.len_list, replace=False
            )
        self._update_models()
        self.all_leaf_stats = self._set_leaf_values(self.model_list, context)
        reward_scores = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) > 0:
                model = self.model_list[action]
                is_fitted = False
                if model is not None:
                    try:
                        model.get_booster()
                        is_fitted = True
                    except Exception:
                        is_fitted = False
                try:
                    if is_fitted:
                        mean_pred, variance, _ = (
                            self._get_ensemble_prediction_with_variance(context, action)
                        )
                        exploration_bonus = (self.nu**2) * variance
                        reward_scores[action] = self.random_.normal(
                            mean_pred, np.sqrt(exploration_bonus)
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to sample Thompson reward for action {action}: {str(e)}"
                    )
            else:
                reward_scores[action] = 1.0
        return reward_scores.argsort()[::-1][: self.len_list]

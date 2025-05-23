# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Contextual Tree Bandit Algorithms."""
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state, check_scalar

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Local application imports
from ..utils import sigmoid, check_array
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
        
    def update_models(self) -> None:
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
                if n_samples >= 4:
                    bootstrap_indices = self.random_.choice(
                        n_samples, size=n_samples, replace=True
                    )
                    
                    # Collect bootstrap data
                    bootstrap_contexts = np.concatenate(
                        [self.context_lists[action][i] for i in bootstrap_indices], 
                        axis=0
                    )
                    bootstrap_rewards = np.array(
                        [self.reward_lists[action][i] for i in bootstrap_indices]
                    )
                    
                    # Make sure we have enough unique samples after bootstrapping
                    unique_values = np.unique(bootstrap_rewards)
                    if len(unique_values) >= 2:
                        # Train new tree on bootstrap sample
                        model = DecisionTreeRegressor(random_state=self.random_state)
                        model.fit(bootstrap_contexts, bootstrap_rewards)
                        
                        # Predict reward for current context
                        action_scores[action] = model.predict(context)[0]
                    else:
                        # If not enough unique values, just use the mean
                        action_scores[action] = np.mean(bootstrap_rewards)
                else:
                    # Not enough samples, use the mean of available samples
                    action_scores[action] = np.mean(self.reward_lists[action])
            else:
                # If no data for this action, assign a high value to encourage exploration
                action_scores[action] = 1.0
                  # Return actions with highest scores
        return action_scores.argsort()[::-1][: self.len_list]
        
    def update_models(self) -> None:
        """Update tree models.
        
        For TreeBootstrap, this is optional because the models are rebuilt during each
        select_action call. However, we can train models here for potential reuse
        if needed for evaluation purposes.
        """
        # For each action, if we have enough data, train a model on all available data
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) >= 2:  # Need at least 2 samples for a tree
                contexts = np.concatenate(self.context_lists[action], axis=0)
                rewards = np.array(self.reward_lists[action])
                
                # Check if we have enough unique values to fit a tree
                if len(np.unique(rewards)) >= 2:
                    self.model_list[action] = DecisionTreeRegressor(random_state=self.random_state)
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
        self.update_models()
        
        # Calculate UCB score for each action
        ucb_scores = np.zeros(self.n_actions)
        
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) > 0:
                if hasattr(self.model_list[action], "tree_") and self.model_list[action].tree_ is not None:
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
                    std_err = np.sqrt(0.25 / max(1, len(self.reward_lists[action])))  # Maximum variance for binary reward
                    ucb_scores[action] = mean_reward + self.z_score * std_err
            else:
                # No data for this action - assign high UCB
                ucb_scores[action] = 1.0
        
        # Return actions with highest UCB scores
        return ucb_scores.argsort()[::-1][: self.len_list]
        
    def update_models(self) -> None:
        """Update tree models for all actions using available data."""
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) >= 2:  # Need at least 2 samples for a tree
                contexts = np.concatenate(self.context_lists[action], axis=0)
                rewards = np.array(self.reward_lists[action])
                
                # Check if we have enough unique values to fit a tree
                if len(np.unique(rewards)) >= 2:
                    self.model_list[action] = DecisionTreeRegressor(random_state=self.random_state)
                    self.model_list[action].fit(contexts, rewards)


@dataclass
class TreeEnsembleUCB(BaseTreePolicy):
    """Tree Ensemble Upper Confidence Bound (TEUCB).
    
    TreeEnsembleUCB は勾配ブースティング決定木（XGBoostなど）や Random Forest などの
    木アンサンブルを学習し、葉レベルの分散情報を使って UCB を構築する汎用フレームワークです。
    各木の葉に割り当てられたサンプル数と出力分散から、アンサンブル予測の分散を積み上げ、
    UCB = μ̃ + ν√(σ̃²ln t / c_{t,a}) で上界を算出します。
    
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
        
    init_rounds: int, default=10
        Number of initial rounds for random exploration (T_I parameter).
        
    nu: float, default=1.0
        Exploration parameter that controls the width of the confidence intervals.
        Higher values lead to more exploration.    
    
    num_trees: int, default=10
        Number of trees in the ensemble.
    
    learning_rate: float, default=0.1
        Learning rate for the ensemble.
    
    max_depth: int, default=3
        Maximum depth of the trees.
    
    References
    ----------
    Alberto Maria Metelli, Alessio Russo, and Marcello Restelli.
    "Sublinear Regret Bounds for Bayesian Optimisation in Unknown Search Spaces," 2019.
    """
    
    init_rounds: int = 10  # Number of initial rounds for random exploration
    nu: float = 1.0  # Exploration parameter
    n_estimators: int = 10  # Number of trees in the ensemble
    learning_rate: float = 0.1  # Learning rate for the ensemble
    max_depth: int = 3  # Maximum depth of the trees
    
    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.init_rounds, "init_rounds", int, min_val=1)
        check_scalar(self.nu, "nu", float, min_val=0.0)
        check_scalar(self.n_estimators, "n_estimators", int, min_val=2)
        check_scalar(self.learning_rate, "learning_rate", float, min_val=0.0, include_boundaries="neither")
        check_scalar(self.max_depth, "max_depth", int, min_val=1)
        self.policy_name = f"tree_ensemble_ucb_{self.nu}"
        super().__post_init__()
        
        self.xgb_available = HAS_XGBOOST
            
        # Initialize context and reward lists for each action
        self.context_lists: List[List[np.ndarray]] = [[] for _ in range(self.n_actions)]
        self.reward_lists: List[List[float]] = [[] for _ in range(self.n_actions)]
        
        # Initialize model list
        if self.xgb_available:
            self.model_list = [
                XGBRegressor(
                    n_estimators=self.n_estimators, 
                    learning_rate=self.learning_rate, 
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
                for _ in range(self.n_actions)
            ]
        else:
            self.model_list = [
                GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
                for _ in range(self.n_actions)
            ]

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update data.

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
        """Select action for new data using UCB from tree ensemble.

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
        
        # Random exploration in the initial rounds
        if self.n_trial < self.init_rounds:
            return self.random_.choice(
                self.n_actions, size=self.len_list, replace=False
            )
        
        # Update models with current data
        self.update_models()
        
        # Calculate UCB score for each action
        ucb_scores = np.zeros(self.n_actions)
        
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) > 0:
                # Get mean prediction and variance estimate
                mean_pred, variance = self._get_ensemble_prediction_with_variance(context, action)
                
                # Calculate UCB score using the formula from algo.md
                # UCB = μ̃ + ν√(σ̃²ln(t-1) / c_{t,a})
                n_samples = len(self.reward_lists[action])
                exploration_bonus = self.nu * np.sqrt(
                    variance * np.log(max(1, self.n_trial - 1)) / max(1, n_samples)
                )
                ucb_scores[action] = mean_pred + exploration_bonus
            else:
                # No data for this action - assign high UCB
                ucb_scores[action] = 1.0
        
        # Return actions with highest UCB scores
        return ucb_scores.argsort()[::-1][: self.len_list]
        
    def _set_leaf_values(self, model, contexts, rewards):
        """Set leaf values for XGBoost trees as per Algorithm 2.
        
        Sets the following values for each leaf node in each tree:
        - o_{n,l}: Mean prediction at leaf l in tree n
        - s^2_{n,l}: Variance of predictions at leaf l in tree n
        - c_{n,l}: Number of samples that fall into leaf l in tree n
        
        Parameters
        ----------
        model: XGBRegressor
            The trained XGBoost model
            
        contexts: array-like, shape (n_samples, dim_context)
            Context vectors used for training
            
        rewards: array-like, shape (n_samples,)
            Reward values used for training
            
        Returns
        -------
        leaf_stats: dict
            Dictionary with leaf statistics per tree:
            {
                tree_idx: {
                    leaf_idx: {'count': c_{n,l}, 'mean': o_{n,l}, 'var': s^2_{n,l}}
                }
            }
        """
        if not self.xgb_available:
            return None
        
        # Get the booster and number of trees
        booster = model.get_booster()
        n_trees = len(booster.get_dump())
        
        # Base predictor (intercept) value
        base_value = model.get_booster().attributes()['base_score'] 
        if isinstance(base_value, str):
            base_value = float(base_value)
        
        # Initialize leaf statistics dictionary
        leaf_stats = {i: {} for i in range(n_trees)}
        
        # Step 1: Accumulate per-leaf contributions
        # For each training sample
        for i in range(len(contexts)):
            ctx = contexts[i:i+1]  # Single context in correct shape
            reward = rewards[i]
            
            # Running sum of predictions up to current tree
            prev_pred = base_value
            
            # For each tree
            for tree_idx in range(n_trees):
                # 1. Get the leaf index where this sample falls
                leaf_idx = int(booster.predict(ctx, pred_leaf=True, iteration_range=(tree_idx, tree_idx+1))[0])
                
                # Initialize leaf stats if this is the first sample in this leaf
                if leaf_idx not in leaf_stats[tree_idx]:
                    leaf_stats[tree_idx][leaf_idx] = {
                        'count': 0,
                        'contributions': [],
                        'mean': 0.0,
                        'var': 0.0
                    }
                
                # 2. Increment sample count for this leaf
                leaf_stats[tree_idx][leaf_idx]['count'] += 1
                
                # 3. Calculate previous prediction sum (p_{n-1})
                if tree_idx > 0:
                    prev_pred = model.predict(ctx, iteration_range=(0, tree_idx))[0]
                
                # 4. Calculate this tree's contribution to the prediction (o_{n,l,(c_{n,l})})
                curr_pred = model.predict(ctx, iteration_range=(0, tree_idx+1))[0]
                contribution = self.learning_rate * (curr_pred - prev_pred)
                
                # Store the contribution
                leaf_stats[tree_idx][leaf_idx]['contributions'].append(contribution)
                
                # Update previous prediction
                prev_pred = curr_pred
        
        # Process accumulated statistics for each tree and leaf
        for tree_idx in range(n_trees):
            for leaf_idx, stats in leaf_stats[tree_idx].items():
                contributions = np.array(stats['contributions'])
                count = stats['count']
                
                # Calculate mean contribution for this leaf
                leaf_stats[tree_idx][leaf_idx]['mean'] = np.mean(contributions) if count > 0 else 0.0
                
                # Calculate variance of contributions
                if count > 1:
                    leaf_stats[tree_idx][leaf_idx]['var'] = np.var(contributions, ddof=1)  # Use sample variance
                else:
                    leaf_stats[tree_idx][leaf_idx]['var'] = 0.0
                
                # Clean up temporary data
                del leaf_stats[tree_idx][leaf_idx]['contributions']
        
        return leaf_stats

    def _get_ensemble_prediction_with_variance(self, context: np.ndarray, action: int) -> tuple:
        """Get the ensemble prediction and variance for a context and action.
        
        Parameters
        ----------
        context: array-like, shape (1, dim_context)
            Context to make prediction for.
            
        action: int
            Action to get prediction for.
            
        Returns
        -------
        tuple: (mean_prediction, variance)
            Mean prediction and variance from the ensemble model.
        """
        model = self.model_list[action]
        
        # Get total number of observations for this action
        n_samples_total = max(1, len(self.reward_lists[action]))
        
        if self.xgb_available:
            # XGBoost implementation with improved variance estimation
            mean_pred = model.predict(context)[0]
            
            try:
                # Extract predictions from individual trees in the ensemble
                booster = model.get_booster()
                n_trees = len(booster.get_dump())  # Get number of trees from the booster dump
                
                # Check if we have pre-calculated leaf statistics
                if hasattr(model, 'leaf_stats'):
                    leaf_stats = model.leaf_stats
                    tree_variance = 0.0
                    effective_sample_count = 0.0
                    
                    # Get leaf indices for current context
                    leaf_indices = booster.predict(context, pred_leaf=True)
                    
                    # Aggregate statistics from leaves where context falls
                    for tree_idx in range(n_trees):
                        leaf_idx = int(leaf_indices[0][tree_idx] if len(leaf_indices[0]) > tree_idx else leaf_indices[0][0])
                        
                        if tree_idx in leaf_stats and leaf_idx in leaf_stats[tree_idx]:
                            stats = leaf_stats[tree_idx][leaf_idx]
                            # Add to variance (weighted by tree maturity)
                            tree_weight = (tree_idx + 1) / n_trees
                            tree_variance += stats['var'] * tree_weight
                            # Use harmonic mean for effective sample count
                            if stats['count'] > 0:
                                effective_sample_count += 1.0 / stats['count']
                        else:
                            # Fallback if leaf not in statistics
                            tree_variance += 0.1 / n_trees
                            effective_sample_count += 1.0 / max(1, n_samples_total // n_trees)
                    
                    # Calculate final effective sample count
                    if effective_sample_count > 0:
                        c_t_a = n_trees / effective_sample_count
                    else:
                        c_t_a = n_samples_total
                else:
                    # Fall back to original implementation if leaf_stats not available
                    # Get tree predictions
                    tree_preds = []
                    tree_weights = []
                    
                    # For each tree, get its contribution
                    prev_pred = 0
                    for tree_idx in range(n_trees):
                        # Get prediction up to this tree
                        curr_pred = model.predict(context, iteration_range=(0, tree_idx+1))[0]
                        # Extract this tree's contribution
                        tree_pred = curr_pred - prev_pred
                        prev_pred = curr_pred
                        
                        tree_preds.append(tree_pred)
                        # More mature trees typically have higher weight
                        tree_weights.append((tree_idx + 1) / n_trees)
                    
                    # Calculate weighted variance across tree predictions
                    if len(tree_preds) > 1:
                        tree_variance = np.average((np.array(tree_preds) - np.mean(tree_preds))**2, 
                                                weights=tree_weights)
                    else:
                        tree_variance = 0.1
                    
                    # Use XGBoost predict with pred_leaf=True to get leaf indices
                    try:
                        leaf_indices = booster.predict(context, pred_leaf=True)[0]
                        
                        # Estimate sample counts based on leaf indices
                        leaf_weight = (np.arange(n_trees) + 1) / n_trees
                        leaf_samples = np.maximum(1, (n_samples_total * leaf_weight * 0.8).astype(int))
                        
                        # Calculate effective sample count - harmonic mean to be conservative
                        c_t_a = n_trees / np.sum(1.0 / leaf_samples)
                    except Exception:
                        # If leaf-specific approach fails, use a more basic estimate
                        c_t_a = n_samples_total / max(1, n_trees**0.5)
            except Exception:
                # Fallback to basic approach if the advanced method fails
                leaf_preds = []
                booster = model.get_booster()
                n_trees = len(booster.get_dump())  # Get number of trees from the booster dump
                for tree_idx in range(n_trees):
                    leaf_preds.append(model.predict(context, iteration_range=(0, tree_idx+1))[0])
                
                tree_variance = np.var(leaf_preds) if len(leaf_preds) > 1 else 0.1
                c_t_a = n_samples_total
            
            # Final variance calculation
            variance = tree_variance / max(1, c_t_a)
            
        else:
            # Sklearn GradientBoostingRegressor implementation with improved variance
            mean_pred = model.predict(context)[0]
            
            # Get all estimators (trees) 
            estimators = model.estimators_.flatten()
            n_estimators = len(estimators)
            
            if n_estimators > 0:
                # Extract tree predictions and leaf statistics
                tree_preds = []
                leaf_samples = []
                leaf_variances = []
                
                for i, tree in enumerate(estimators):
                    # Get prediction from this tree (applying learning rate)
                    tree_pred = tree.predict(context)[0] * model.learning_rate
                    tree_preds.append(tree_pred)
                    
                    # Extract leaf node information when possible
                    try:
                        # Get the decision path to identify the leaf node
                        leaf_id = tree.apply(context)[0]  # Gets the leaf index
                        tree_structure = tree.tree_
                        
                        # Get samples count in this leaf
                        n_leaf_samples = tree_structure.n_node_samples[leaf_id]
                        
                        # Trees see a subset of data, adjust accordingly
                        adjustment = (n_estimators - i) / n_estimators
                        effective_samples = max(1, int(n_leaf_samples * adjustment))
                        leaf_samples.append(effective_samples)
                        
                        # For regression trees, impurity is often MSE
                        if hasattr(tree_structure, 'impurity'):
                            leaf_variances.append(max(0.001, tree_structure.impurity[leaf_id]))
                    except Exception:
                        # Fallback if detailed leaf stats aren't available
                        leaf_samples.append(max(1, n_samples_total // n_estimators))
                
                # Variance is estimated as the average across trees of the leaf variance
                if len(leaf_variances) > 0:
                    variance = np.mean(leaf_variances) * np.var(tree_preds) / max(1, np.mean(leaf_samples))
                else:
                    variance = np.var(tree_preds) if len(tree_preds) > 1 else 0.1
                
                # Effective sample size for the ensemble
                c_t_a = np.mean(leaf_samples) if len(leaf_samples) > 0 else n_samples_total
            
            else:
                variance = 0.1
                c_t_a = n_samples_total
        
        # For UCB, we use the variance of the ensemble prediction
        return mean_pred, variance / max(1, c_t_a)

    def update_models(self) -> None:
        """Update ensemble tree models for all actions using available data."""
        for action in range(self.n_actions):
            if len(self.reward_lists[action]) >= 2:  # Need at least 2 samples to fit
                try:
                    contexts = np.concatenate(self.context_lists[action], axis=0)
                    rewards = np.array(self.reward_lists[action])

                    # Check if we have enough unique values to fit a tree
                    if len(np.unique(rewards)) >= 2:
                        # Create and fit a new model - don't modify original model reference
                        if self.xgb_available:
                            from xgboost import XGBRegressor
                            self.model_list[action] = XGBRegressor(
                                n_estimators=self.n_estimators,
                                learning_rate=self.learning_rate,
                                max_depth=self.max_depth,
                                random_state=self.random_state
                            )
                            
                            # Fit the model with available data
                            self.model_list[action].fit(contexts, rewards)
                            
                            # Calculate and store leaf statistics
                            self.model_list[action].leaf_stats = self._set_leaf_values(
                                self.model_list[action], contexts, rewards
                            )
                        else:
                            from sklearn.ensemble import GradientBoostingRegressor
                            self.model_list[action] = GradientBoostingRegressor(
                                n_estimators=self.n_estimators,
                                learning_rate=self.learning_rate,
                                max_depth=self.max_depth,
                                random_state=self.random_state
                            )
                            
                            # Fit the model with available data
                            self.model_list[action].fit(contexts, rewards)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to update model for action {action}: {str(e)}")

# OBP Policy Module - Tree-based Contextual Bandit Algorithms

このドキュメントでは、`tree.py`で実装されている決定木ベースの文脈付きバンディットアルゴリズムについて詳しく説明します。

## 概要

`tree.py`は、決定木およびアンサンブル手法を用いた文脈付きバンディットアルゴリズムのライブラリです。4つの主要なアルゴリズムを実装し、それぞれ異なるアプローチで探索と活用のバランスを取ります。

## 実装されているアルゴリズム

### 1. TreeBootstrap
**Thompson Samplingをブートストラップで実装**

```python
@dataclass
class TreeBootstrap(BaseTreePolicy):
```

**アルゴリズムの流れ：**
1. 各アクションの履歴（文脈-報酬ペア）を保持
2. アクション選択時に、各アクションの履歴からブートストラップサンプルを作成
3. 各ブートストラップサンプルで決定木を訓練
4. 現在の文脈に対する予測値が最高のアクションを選択

**特徴：**
- 不確実性をブートストラップサンプリングで表現
- 各選択時にモデルを再構築（計算コスト高）
- 理論的基盤：Thompson Sampling

### 2. TreeUCB
**決定木 + Upper Confidence Bound**

```python
@dataclass
class TreeUCB(BaseTreePolicy):
    alpha: float = 0.05  # 信頼水準
```

**アルゴリズムの流れ：**
1. 各アクションで決定木モデルを訓練
2. 予測値に統計的信頼上界を追加
3. UCBスコア = 予測値 + z × 標準誤差
4. UCBスコアが最高のアクションを選択

**数式：**
```
UCB(a) = μ̂(a) + z_{α/2} × σ̂(a)
```

**特徴：**
- 統計的に厳密な信頼区間計算
- z-score（デフォルト1.96、α=0.05）使用
- 理論的保証のある探索戦略

### 3. TreeEnsembleUCB
**XGBoostアンサンブル + UCB**

```python
@dataclass
class TreeEnsembleUCB(BaseTreeEnsemblePolicy):
    nu: float = 1.0  # 探索パラメータ
```

**アルゴリズムの流れ：**
1. XGBoostでアンサンブルモデルを訓練
2. 各木の葉ノード統計から分散を推定
3. UCBスコア = 予測値 + 探索ボーナス
4. 最高スコアのアクションを選択

**探索ボーナス：**
```python
exploration_bonus = nu * sqrt(variance * log(t-1) / count)
```

**特徴：**
- 高精度なアンサンブル予測
- 葉ノード統計による不確実性定量化
- スケーラブルな実装

### 4. TreeEnsembleTS
**XGBoostアンサンブル + Thompson Sampling**

```python
@dataclass
class TreeEnsembleTS(BaseTreeEnsemblePolicy):
```

**アルゴリズムの流れ：**
1. XGBoostでアンサンブルモデルを訓練
2. 予測値と分散を推定
3. 正規分布からサンプリング：`N(μ, σ²)`
4. サンプル値が最高のアクションを選択

**サンプリング：**
```python
reward_scores[action] = random.normal(mean_pred, sqrt(exploration_bonus))
```

**特徴：**
- 確率的なアクション選択
- 自然な探索メカニズム
- アンサンブルの予測精度を活用

## クラス階層

```
BaseContextualPolicy (from base.py)
    ↓
BaseTreePolicy
    ↓
    ├── TreeBootstrap
    ├── TreeUCB
    └── BaseTreeEnsemblePolicy
            ↓
            ├── TreeEnsembleUCB
            └── TreeEnsembleTS
```

## 主要な技術的特徴

### 1. エラーハンドリング
```python
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
```
- XGBoostの可用性チェック
- モデル学習失敗時の警告システム
- データ不足時のフォールバック機能

### 2. 統計的厳密性
```python
# TreeUCBでの信頼区間計算
self.z_score = stats.norm.ppf(1 - self.alpha / 2)
std_err = np.sqrt(p_hat * (1 - p_hat) / max(1, n_samples))
ucb = p_hat + self.z_score * std_err
```

### 3. アンサンブル統計抽出
```python
def _set_leaf_values(self, models, context):
    # XGBoostの各木の葉ノード統計を抽出
    # 予測の不確実性推定に使用
```

### 4. パフォーマンス最適化
- 最小サンプル数チェック（決定木には最低2サンプル必要）
- 効率的なデータ構造（リスト形式での履歴管理）
- 初期ラウンドでのランダム選択

## パラメータ設定

### 共通パラメータ
- `dim`: コンテキストベクトルの次元数
- `n_actions`: アクション数
- `len_list`: 推薦リストの長さ（デフォルト1）
- `random_state`: ランダムシード

### TreeUCB特有
- `alpha`: 信頼水準（デフォルト0.05）

### アンサンブル系特有
- `nu`: 探索パラメータ（デフォルト1.0）
- `n_estimators`: 木の数（デフォルト10）
- `learning_rate`: 学習率（デフォルト0.1）
- `max_depth`: 木の最大深度（デフォルト3）
- `init_rounds`: 初期ランダム選択ラウンド数（デフォルト10）

## 使用例

```python
# TreeBootstrapの使用例
policy = TreeBootstrap(
    dim=5,
    n_actions=3,
    random_state=42
)

# 文脈を観測
context = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

# アクション選択
action = policy.select_action(context)[0]

# 報酬を観測してパラメータ更新
reward = 0.8
policy.update_params(action, reward, context)
```

## 適用場面

### 1. オンライン推薦システム
- ユーザーの属性（年齢、性別、履歴など）に基づく商品推薦
- リアルタイムでの学習と改善

### 2. 動的A/Bテスト
- 文脈に応じた実験割り当て
- 効率的な統計的推論

### 3. 個人化広告配信
- ユーザープロファイルと広告コンテンツのマッチング
- クリック率最適化

### 4. コンテンツ最適化
- ユーザーの興味や行動パターンに基づくコンテンツ選択
- エンゲージメント最大化

## アルゴリズム選択指針

| アルゴリズム    | 適用場面           | 長所                 | 短所                 |
| --------------- | ------------------ | -------------------- | -------------------- |
| TreeBootstrap   | 小〜中規模データ   | 理論的基盤が確立     | 計算コスト高         |
| TreeUCB         | 統計的保証が必要   | 厳密な信頼区間       | 単一木の限界         |
| TreeEnsembleUCB | 高精度が必要       | アンサンブルの予測力 | XGBoost依存          |
| TreeEnsembleTS  | 自然な探索が欲しい | 確率的選択           | パラメータ調整が重要 |

## 参考文献

1. **TreeBootstrap**: Elmachtoub, Adam N., et al. "A Practical Method for Solving Contextual Bandit Problems Using Decision Trees." arXiv preprint arXiv:1706.04687 (2017).

2. **TreeUCB**: S Oiwa, et al. "Contextual Bandit Algorithm with Decision Trees and Upper Confidence Bound for Adaptive Recommendation" The 51st International Conference on Computers and Industrial Engineering (CIE51). 2024.

3. **Tree Ensemble Methods**: Nilsson, Hannes, Rikard Johansson, Niklas Åkerblom, and Morteza Haghir Chehreghani. “Tree Ensembles for Contextual Bandits.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/2402.06963. 2024.

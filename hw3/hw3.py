"""
mini_autograd.py — 輕量級反向傳播框架與神經網路組件

包含：
  - Var: 支援自動微分的純量節點
  - AdamOptimizer: 具備學習率衰減機制的 Adam 演算法
  - dense_forward: 全連接層矩陣運算
  - safe_softmax: 具備防溢位機制的 Softmax
  - root_mean_square_norm: RMSNorm 正規化
"""

import math

class Var:
    """純 Python 實作的自動微分純量，支援建構計算圖與計算梯度。"""
    __slots__ = ('val', 'g', '_deps', '_partials')

    def __init__(self, val, deps=(), partials=()):
        self.val = val
        self.g = 0.0
        self._deps = deps
        self._partials = partials

    def __add__(self, rhs):
        rhs = rhs if isinstance(rhs, Var) else Var(rhs)
        return Var(self.val + rhs.val, (self, rhs), (1.0, 1.0))

    def __mul__(self, rhs):
        rhs = rhs if isinstance(rhs, Var) else Var(rhs)
        return Var(self.val * rhs.val, (self, rhs), (rhs.val, self.val))

    def __pow__(self, p):
        return Var(self.val ** p, (self,), (p * (self.val ** (p - 1)),))

    def log(self):
        return Var(math.log(self.val), (self,), (1.0 / self.val,))

    def exp(self):
        return Var(math.exp(self.val), (self,), (math.exp(self.val),))

    def relu(self):
        return Var(max(0.0, self.val), (self,), (1.0 if self.val > 0 else 0.0,))

    # 運算子多載的簡化映射
    def __neg__(self):         return self * -1
    def __radd__(self, rhs):   return self + rhs
    def __sub__(self, rhs):    return self + (-rhs)
    def __rsub__(self, rhs):   return rhs + (-self)
    def __rmul__(self, rhs):   return self * rhs
    def __truediv__(self, rhs):  return self * (rhs ** -1)
    def __rtruediv__(self, rhs): return rhs * (self ** -1)

    def backward(self):
        """觸發反向傳播，依照拓撲排序計算計算圖中所有節點的梯度。"""
        order = []
        seen = set()

        def walk(node):
            if node not in seen:
                seen.add(node)
                for dep in node._deps:
                    walk(dep)
                order.append(node)
        
        walk(self)
        self.g = 1.0  # 起始梯度設為 1

        # 逆向走訪計算圖，將梯度分配給子節點
        for node in reversed(order):
            for dep, partial_deriv in zip(node._deps, node._partials):
                dep.g += partial_deriv * node.g

    def __repr__(self):
        return f"Var(val={self.val:.4f}, g={self.g:.4f})"


class AdamOptimizer:
    """Adam 參數優化器，允許在每次 step 覆寫學習率以實現退火。"""

    def __init__(self, parameters, learning_rate=0.01, b1=0.85, b2=0.99, epsilon=1e-8):
        self.parameters = parameters
        self.base_lr = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        # 初始化一階與二階動量
        self.moment1 = [0.0] * len(parameters)
        self.moment2 = [0.0] * len(parameters)
        self.t = 0

    def step(self, current_lr=None):
        """更新所有註冊的參數，並在結束後歸零梯度。"""
        self.t += 1
        active_lr = current_lr if current_lr is not None else self.base_lr
        
        for idx, param in enumerate(self.parameters):
            # 更新動量估計
            self.moment1[idx] = self.b1 * self.moment1[idx] + (1 - self.b1) * param.g
            self.moment2[idx] = self.b2 * self.moment2[idx] + (1 - self.b2) * (param.g ** 2)
            
            # 偏差校正
            m_bias_corr = self.moment1[idx] / (1 - self.b1 ** self.t)
            v_bias_corr = self.moment2[idx] / (1 - self.b2 ** self.t)
            
            # 權重更新
            param.val -= active_lr * m_bias_corr / (math.sqrt(v_bias_corr) + self.epsilon)
            
            # 清除梯度
            param.g = 0.0


def dense_forward(x, weight_matrix):
    """執行全連接層的前向傳播：y = W @ x"""
    return [sum(w_i * x_i for w_i, x_i in zip(row, x)) for row in weight_matrix]


def safe_softmax(logits):
    """具備 Log-Sum-Exp 減法技巧的穩定版 Softmax。"""
    c = max(v.val for v in logits)
    exp_vals = [(v - c).exp() for v in logits]
    sum_exp = sum(exp_vals)
    return [e / sum_exp for e in exp_vals]


def root_mean_square_norm(features):
    """RMSNorm 實作，效能優於傳統 LayerNorm 且不需 mean shift。"""
    mean_sq = sum(f * f for f in features) / len(features)
    inv_std = (mean_sq + 1e-5) ** -0.5
    return [f * inv_std for f in features]


def train_step(model, optim, seq_tokens, current_step, total_steps):
    """
    處理單一 Batch 的訓練流程：前向傳播 -> 計算損失 -> 反向傳播 -> 更新參數。
    """
    seq_len = min(model.block_size, len(seq_tokens) - 1)
    k_cache = [[] for _ in range(model.n_layer)]
    v_cache = [[] for _ in range(model.n_layer)]

    step_losses = []
    for pos in range(seq_len):
        x_tok, y_tok = seq_tokens[pos], seq_tokens[pos + 1]
        
        out_logits = model(x_tok, pos, k_cache, v_cache)
        probabilities = safe_softmax(out_logits)
        
        # 針對目標類別計算 NLL Loss
        loss_at_t = -probabilities[y_tok].log()
        step_losses.append(loss_at_t)
        
    # 計算平均 Loss
    avg_loss = (1.0 / seq_len) * sum(step_losses)

    # 反向傳播
    avg_loss.backward()

    # 計算當前學習率並更新權重
    annealed_lr = optim.base_lr * (1.0 - current_step / total_steps)
    optim.step(current_lr=annealed_lr)

    return avg_loss.val

def basic_ce_loss(logits, label_id):
    """
    最直觀的交叉熵實作，直接依賴 softmax 取得機率後取 log。
    缺點是容易遇到機率趨近於 0 時的數值不穩定問題。
    """
    preds = safe_softmax(logits)
    return -preds[label_id].log()

"""
在實務上的深度學習函式庫（例如 PyTorch 內建的 F.cross_entropy）中，
通常會將 Softmax 與負對數概似（NLL）合併計算，而不是分開執行。
這是因為如果在機率運算階段產生了極小值，直接取 log 會導致 NaN 或 -inf 的崩潰。

底層會利用對數法則將數學式展開：

$$-\log\left(\frac{e^{x_t}}{\sum e^{x_i}}\right) = \log\left(\sum e^{x_i}\right) - x_t$$

結合減去最大值的平移技巧（Log-Sum-Exp Trick），
我們可以將交叉熵改寫成以下更精簡且完全防溢位的版本：
"""
def fused_cross_entropy(logits, label_id):
    """
    高穩定性的 Cross Entropy 函式。
    透過融合 Log-Sum-Exp 技巧，避免了除法運算與對趨近於零的數值取 Log。
    """
    # 1. 取得最大值 (防溢位平移量)
    c = max(v.val for v in logits)
    
    # 2. 針對平移後的 logits 計算 exp
    shifted_exps = [(v - c).exp() for v in logits]
    
    # 3. 建立 Log-Sum 節點
    lse = sum(shifted_exps).log()
    
    # 4. 直接套用展開公式計算 Loss: log(sum(exp(x-c))) - (x_label - c)
    target_val_shifted = logits[label_id] - c
    
    return lse - target_val_shifted

"""
以最原子化、純 Python 且零依賴的方式，進行 GPT 的訓練與推論。
本程式碼參考自 Andrej Karpathy 的 MicroGPT。
"""
import os
import math
import random

random.seed(42) # 在混沌中建立秩序

# =============================================================================
# 1. 資料集與分詞器 (Dataset & Tokenizer)
# =============================================================================
# 假設我們有一個文件列表 `docs`（例如名字列表）
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"文件數量: {len(docs)}")

# 建立分詞器將字串轉換為整數序列 (tokens)
uchars = sorted(set(''.join(docs))) # 資料集中不重複的字元成為 token id (0..n-1)
BOS = len(uchars) # 特殊序列起始標記 (Beginning of Sequence) 的 token id
vocab_size = len(uchars) + 1 # 總詞彙量（包含 BOS）
print(f"詞彙表大小: {vocab_size}")

# =============================================================================
# 2. 自動微分引擎 (Autograd Engine)
# =============================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    
    def __init__(self, data, children=(), local_grads=()):
        self.data = data           # 節點的純量數值（前向傳播時計算）
        self.grad = 0              # 損失函數對此節點的梯度（反向傳播時計算）
        self._children = children  # 計算圖中的子節點
        self._local_grads = local_grads # 對子節點的局部梯度
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
        
    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
        
    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))
        
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
        
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
        
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1 # 設定最終輸出的初始梯度
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# =============================================================================
# 3. 初始化模型參數 (Model Parameters)
# =============================================================================
n_layer = 1      # Transformer 網路深度 (層數)
n_embd = 16      # 網路寬度 (嵌入維度)
block_size = 16  # 注意力視窗的最大上下文長度
n_head = 4       # 注意力頭的數量
head_dim = n_embd // n_head # 每個頭的維度

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row] # 攤平為列表
print(f"模型參數總量: {len(params)}")

# =============================================================================
# 4. 模型架構定義 (Model Architecture: GPT-2 variant)
# =============================================================================
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # Token 嵌入
    pos_emb = state_dict['wpe'][pos_id]   # 位置嵌入
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # 組合
    x = rmsnorm(x)
    
    for li in range(n_layer):
        # 1) 多頭注意力機制 (Multi-head Attention)
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # 點積注意力運算
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
            
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # 2) MLP 模組
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
        
    logits = linear(x, state_dict['lm_head'])
    return logits

# =============================================================================
# 5. 訓練迴圈 (Training Loop with Adam Optimizer)
# =============================================================================
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # 一階動量
v = [0.0] * len(params) # 二階動量

num_steps = 1000 # 訓練步數

for step in range(num_steps):
    # 準備一個單一文件，首尾加上 BOS 標記
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    
    # 建立計算圖直到求出損失 (Loss)
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log() # 交叉熵損失 (Cross-Entropy)
        losses.append(loss_t)
        
    loss = (1 / n) * sum(losses) # 整個序列的平均損失
    
    # 反向傳播計算梯度
    loss.backward()
    
    # Adam 最佳化器更新參數
    lr_t = learning_rate * (1 - step / num_steps) # 線性學習率衰減
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0 # 梯度歸零清空
        
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# =============================================================================
# 6. 模型推論 (Inference Generation)
# =============================================================================
temperature = 0.5 # 數值越低越保守，數值越高越隨機
print("\n--- inference (幻覺生成的新名字) ---")

for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        
        # 根據機率隨機選擇下一個 token
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: 
            break
        sample.append(uchars[token_id])
        
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

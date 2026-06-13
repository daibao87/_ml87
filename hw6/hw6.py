import random
from collections import defaultdict
import jieba

class CompleteMarkovLM:
    def __init__(self):
        # 儲存不同階層的 N-gram 字典
        self.trigram = defaultdict(list)  # 紀錄 (詞A, 詞B) -> 詞C
        self.bigram = defaultdict(list)   # 紀錄 (詞B) -> 詞C (退避用)
        self.unigram = []                 # 紀錄所有出現過的詞 (最終退避用)
        
    def train(self, sentences):
        """
        訓練模型：傳入一個包含多個句子的列表
        """
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 使用 jieba 斷詞，並在頭尾加入 BOS (Begin of Sentence) 與 EOS (End of Sentence)
            words = ['<BOS>'] + list(jieba.cut(sentence)) + ['<EOS>']
            
            # 建立 Unigram 語料庫 (排除特殊標記)
            self.unigram.extend([w for w in words if w not in ('<BOS>', '<EOS>')])
            
            # 建立 Bigram 字典 (用於退避與生成開頭)
            for i in range(len(words) - 1):
                self.bigram[words[i]].append(words[i+1])
                
            # 建立 Trigram 字典 (主力預測模型)
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i+1], words[i+2]
                self.trigram[(w1, w2)].append(w3)
                
    def generate(self, max_len=50):
        """
        生成文本：自動從句首開始，遇到句尾或達到最大長度時停止
        """
        # 從句首開始，利用

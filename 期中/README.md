# 🚀 AI 通訊協定：設計與優化多語言模型協作的「高密度機器原生語言」
(AI Communication Protocol: Designing and Optimizing High-Density Machine-Native Language for Multi-Agent Collaboration)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LLM-Agnostic](https://img.shields.io/badge/LLM-Agnostic-orange.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本專案旨在探索大語言模型（LLM）多智能體（Multi-Agent）協作中的**通訊效率極限**。人類自然語言（如中、英文）雖然對人類直觀，但對於機器而言存在大量的冗餘與資訊密度低下的問題，這會導致高昂的 Token 成本與不必要的延遲。

本專案實作了一個雙智能體協作架構（Coder Agent & Tester Agent），透過 **Harness / Prompt Engineering** 強制 Agent 間發展出人類難以直接閱讀，但**資訊密度極高、語意精準、傳輸快速的「機器原生語言 (Machine-Native Language, MNL)」**。並針對其 Token 消耗量、推論時間與任務成功率進行全方位的 Benchmarking 數據對比。

---

## 📌 項目核心背景與動機

在現行的 Multi-Agent 框架（如 CrewAI, AutoGen）中，Agent 間的對話大多採用自然語言。本專案提出一個假設：**「如果將人類排除在溝通迴圈之外，AI 是否能發展出更高效的機器原生密碼來提升協作效率？」**

### 本專案的核心科學問題：
1. **資訊壓縮能力**：LLM 能否在高度壓縮的符號/代碼中，不失真地傳遞複雜的邏輯與錯誤日誌（Error Logs）？
2. **通訊成本優化**：相較於傳統自然語言，機器原生語言能降低多少比例的 Token 消耗？
3. **可解釋性衝突**：當機器溝通進入「黑盒子狀態」，人類如何在維持高效的同時進行安全監管？

---

## 🏗️ 系統架構圖

本系統主要由三個核心模組組成：**智能體協作框架 (Agent Framework)**、**機器語言編譯引擎 (MNL Protocol Engine)** 以及 **效能基準測試套件 (Benchmarking Suite)**。

```text
+-------------------------------------------------------------------------+
|                           Benchmarking Suite                            |
|    [Tiktoken Evaluator]  <->  [Latency Monitor]  <->  [Accuracy Track]  |
+-------------------------------------------------------------------------+
                                     ^
                                     | (監控通訊)
                                     v
+------------------+     機器原生語言 (MNL)      +------------------+
|   Agent A: Coder | =========================> |  Agent B: Tester |
| (負責編寫與修正代碼) | <========================= | (負責執行與回傳測試) |
+------------------+    高密度符號/Token 壓縮     +------------------+
        ^                                                    ^
        |                                                    |
  [System Prompt]                                      [System Prompt]
  (定義語法生成規則)                                    (定義解碼與回應規範)

import os
import time
import json
import io
import sys
import contextlib
import tiktoken
from openai import OpenAI

# ==========================================
# ⚙️ Part 1: 環境與設定 (Configuration)
# ==========================================
# 請替換為您的 OpenAI API Key，或在環境變數中設定 OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-your-api-key-here")
client = OpenAI()

MODEL_NAME = "gpt-3.5-turbo" # 建議使用 gpt-3.5-turbo 或 gpt-4o-mini 來測試
MAX_TURNS = 3 # Agent 來回除錯的最大輪數

# 測試用的編程任務資料集 (Mock Data)
TASKS = [
    {
        "id": 1,
        "description": "寫一個 Python 函數 `fibonacci(n)`，回傳第 n 個費氏數列數字。當 n=5 時，預期回傳 5。",
        "test_code": "assert fibonacci(5) == 5\nassert fibonacci(10) == 55"
    },
    {
        "id": 2,
        "description": "寫一個 Python 函數 `reverse_string(s)`，反轉字串並回傳。預期 reverse_string('hello') 回傳 'olleh'。",
        "test_code": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('AI') == 'IA'"
    }
]

# ==========================================
# 🧠 Part 2: 協議引擎 (Protocol Engine / Prompts)
# ==========================================
# 對照組：自然語言 (Natural Language)
CODER_NL_PROMPT = """你是一個資深工程師。請根據需求撰寫 Python 程式碼。
如果 Tester 告訴你代碼有錯，請閱讀他用自然語言寫的錯誤報告，並修正代碼。
回覆時，請先用自然語言解釋你的修改，然後附上完整的程式碼。"""

TESTER_NL_PROMPT = """你是一個 QA 測試工程師。
我會提供執行 Python 程式碼後的結果（包含錯誤或成功訊息）。
請用「詳細的自然語言（人類閱讀的語言）」向 Coder 解釋發生了什麼錯誤，告訴他錯誤在哪一行、原因是什麼。
如果成功，請告訴他測試通過。"""

# 實驗組：機器原生語言 (Machine-Native Language, MNL)
CODER_MNL_PROMPT = """[ROLE: AI_CODER]
你現在與 Tester Agent 進行高效協作。禁止使用自然語言閒聊。
如果 Tester 傳回錯誤，它會使用 MNL 格式 (例如: [E:IndexError|L:5|C:arr[i]])。
請直接解析該格式並修正代碼。回應時「只需輸出 Python 程式碼」，不需要任何解釋。"""

TESTER_MNL_PROMPT = """[ROLE: AI_TESTER]
你現在與 Coder Agent 進行高密度通訊。禁止使用人類自然語言。
請根據我提供的代碼執行結果，壓縮出最高密度的「機器原生語言 (MNL)」。
規則：
1. 成功回傳：[STAT:PASS]
2. 錯誤回傳範例：[E:NameError|L:2|V:fibonacci_not_defined] (E=Error Type, L=Line, V=Value/Context)
絕對不可輸出多餘的字元。"""

# ==========================================
# 🛠️ Part 3: 工具套件 (Utils)
# ==========================================
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """計算字串的 Token 數量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

@contextlib.contextmanager
def capture_stdout():
    """攔截 print 與標準輸出"""
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def run_code_sandbox(source_code: str, test_code: str) -> dict:
    """在受限環境中執行代碼 (警告: 僅供實驗用途，請勿執行不明代碼)"""
    combined_code = source_code + "\n" + test_code
    env = {}
    
    with capture_stdout() as (out, err):
        try:
            exec(combined_code, env)
            return {"status": "success", "output": out.getvalue()}
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc(limit=2) # 限制錯誤追蹤深度
            return {"status": "error", "error": str(e), "traceback": error_msg}

def extract_python_code(text: str) -> str:
    """從 LLM 回應中提取 Python 代碼區塊"""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

# ==========================================
# 🤖 Part 4: 智能體框架 (Agent Framework)
# ==========================================
def call_llm(system_prompt: str, user_prompt: str) -> tuple[str, int]:
    """呼叫 LLM 並計算耗時與 Token"""
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content
        latency = time.time() - start_time
        return content, latency
    except Exception as e:
        print(f"LLM API 錯誤: {e}")
        return "", 0

def run_agent_workflow(task: dict, mode: str = "NL"):
    """執行單一任務的 Multi-Agent 協作迴圈"""
    print(f"\n[{mode} 模式] 開始任務: {task['description'][:30]}...")
    
    # 根據模式選擇 Prompt
    coder_sys_prompt = CODER_NL_PROMPT if mode == "NL" else CODER_MNL_PROMPT
    tester_sys_prompt = TESTER_NL_PROMPT if mode == "NL" else TESTER_MNL_PROMPT
    
    metrics = {
        "total_tokens": 0,
        "total_latency": 0.0,
        "turns": 0,
        "success": False
    }
    
    # 初始需求
    current_feedback = f"需求: {task['description']}\n測試代碼:\n{task['test_code']}"
    
    for turn in range(MAX_TURNS):
        metrics["turns"] += 1
        print(f"  ▶ 輪次 {turn + 1}/{MAX_TURNS}")
        
        # 1. Coder Agent 寫代碼
        coder_response, latency = call_llm(coder_sys_prompt, current_feedback)
        metrics["total_latency"] += latency
        metrics["total_tokens"] += count_tokens(coder_sys_prompt + current_feedback + coder_response)
        
        code_to_run = extract_python_code(coder_response)
        
        # 2. 本地沙盒執行
        execution_result = run_code_sandbox(code_to_run, task['test_code'])
        
        if execution_result["status"] == "success":
            print("    ✅ 代碼執行成功！")
            tester_input = "執行成功，沒有報錯。"
        else:
            print("    ❌ 代碼報錯！")
            tester_input = f"錯誤訊息:\n{execution_result['error']}\n追蹤:\n{execution_result['traceback']}"
        
        # 3. Tester Agent 產生回饋 (轉換為對應語言)
        tester_response, latency = call_llm(tester_sys_prompt, tester_input)
        metrics["total_latency"] += latency
        metrics["total_tokens"] += count_tokens(tester_sys_prompt + tester_input + tester_response)
        
        print(f"    [{mode} 傳輸內容]: {tester_response.strip()[:60]}...")
        
        # 檢查是否通過
        if execution_result["status"] == "success" or "[STAT:PASS]" in tester_response or "通過" in tester_response:
            metrics["success"] = True
            break
            
        # 將 Tester 的回饋交給下一輪的 Coder
        current_feedback = f"上次代碼測試失敗。Tester 回報: {tester_response}"

    return metrics

# ==========================================
# 📊 Part 5: 基準測試與報告 (Benchmarking)
# ==========================================
def main():
    print("🚀 啟動 AI 機器原生語言 (MNL) 協議基準測試\n" + "="*50)
    
    # 若未設定真實 API Key，程式會提示並退出
    if client.api_key == "sk-your-api-key-here":
        print("⚠️ 警告：請先替換真實的 OPENAI_API_KEY 才能運行測試！")
        return

    results = []
    
    for task in TASKS:
        # 測試自然語言 (NL)
        nl_metrics = run_agent_workflow(task, mode="NL")
        
        # 測試機器原生語言 (MNL)
        mnl_metrics = run_agent_workflow(task, mode="MNL")
        
        results.append({
            "task_id": task["id"],
            "NL": nl_metrics,
            "MNL": mnl_metrics
        })

    # 生成總結報告
    print("\n" + "="*50)
    print("📈 基準測試結果報告 (Benchmarking Report)")
    print("="*50)
    
    total_nl_tokens = sum(r["NL"]["total_tokens"] for r in results)
    total_mnl_tokens = sum(r["MNL"]["total_tokens"] for r in results)
    
    for r in results:
        tid = r['task_id']
        print(f"任務 {tid}:")
        print(f"  - [NL 模式]  成功: {r['NL']['success']}, 消耗 Tokens: {r['NL']['total_tokens']}, 耗時: {r['NL']['total_latency']:.2f}s")
        print(f"  - [MNL 模式] 成功: {r['MNL']['success']}, 消耗 Tokens: {r['MNL']['total_tokens']}, 耗時: {r['MNL']['total_latency']:.2f}s")
    
    print("-" * 50)
    token_saved = (total_nl_tokens - total_mnl_tokens) / max(total_nl_tokens, 1) * 100
    print(f"🏆 總結：使用機器原生語言 (MNL) 協定，Token 節省率達: {token_saved:.2f}%\n")

if __name__ == "__main__":
    main()

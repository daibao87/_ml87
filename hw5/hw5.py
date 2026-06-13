#!/usr/bin/env python3
# agent0.py - AI Agent with enhanced sandbox and manual authorization (v4-secure)
# Run: python agent0.py

import subprocess
import os
import sys
import re
from mlx_lm import load, generate

# ─── Configuration ───

# 預設工作區，後續會在 main() 中進行實體路徑解析 (realpath)
WORKSPACE = os.path.expanduser("~/.agent0")
MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MAX_TURNS = 5
AUTO_AUTHORIZE = False

# ─── Sandbox & Authorization ───

def resolve_path(path):
    """將路徑解析為絕對的實體路徑，處理 ~、相對路徑，並展開所有符號連結 (Symlink) 以防範繞過攻擊。"""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE, path)
    # 使用 realpath 真正向作業系統確認實體路徑，防止透過符號連結 (Symlink Bypass) 逃脫沙盒
    return os.path.realpath(path)

def is_path_safe(path):
    """檢查解析後的實體路徑是否嚴格位於指定的 WORKSPACE 工作區目錄內。"""
    abs_path = resolve_path(path)
    ws = os.path.realpath(WORKSPACE)
    try:
        return os.path.commonpath([abs_path, ws]) == ws
    except ValueError:
        # 當路徑處於不同的磁碟驅動器（例如 Windows 跨盤符）時，commonpath 會拋出 ValueError
        return False

def authorize(message):
    if AUTO_AUTHORIZE:
        return True
    print(f"\n⚠️  {message}")
    answer = input("是否核可該操作？(y/N): ").strip().lower()
    return answer == "y"

def handle_read_file(path):
    safe = is_path_safe(path)
    if not safe and not authorize(f"LLM 請求讀取外部檔案（工作區外）：{path}"):
        return "Error: Permission denied by user."
    
    target = resolve_path(path)
    if not safe:
        print(f"   [安全警告] 經使用者授權存取外部路徑：{target}")
    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error: {e}"

def handle_write_file(path, content):
    safe = is_path_safe(path)
    if not safe and not authorize(f"LLM 請求寫入外部檔案（工作區外）：{path}"):
        return "Error: Permission denied by user."
    
    target = resolve_path(path)
    if not safe:
        print(f"   [安全警告] 經使用者授權寫入外部路徑：{target}")
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {target}"
    except Exception as e:
        return f"Error: {e}"

def extract_tools(text):
    """從 LLM 回應中提取所有工具標籤，並包容單雙引號的容錯率，保持執行順序。"""
    text = re.sub(r'```\w*\n?', '', text)
    text = text.replace('```', '')
    tools = []
    
    # 1. 提取 Shell 指令
    for m in re.finditer(r'<shell>(.+?)</shell>', text, re.DOTALL):
        tools.append((m.start(), 'shell', m.group(1).strip(), None))
        
    # 2. 提取 讀取檔案（支援 path="..." 與 path='...'）
    for m in re.finditer(r'<read_file\s+path=(["\'])(.*?)\1\s*/?>', text):
        tools.append((m.start(), 'read_file', m.group(2).strip(), None))
        
    # 3. 提取 寫入檔案（支援 path="..." 與 path='...'）
    for m in re.finditer(r'<write_file\s+path=(["\'])(.*?)\1>(.*?)</write_file>', text, re.DOTALL):
        tools.append((m.start(), 'write_file', m.group(2).strip(), m.group(3)))
        
    tools.sort(key=lambda x: x[0])
    return tools

# ─── Memory ───

conversation_history = []
key_info = []

# ─── MLX Model ───

_model = None
_tokenizer = None

def load_model():
    global

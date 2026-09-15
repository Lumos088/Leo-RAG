"""Start the FastAPI backend and Streamlit frontend with one command."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web_ui"


def parse_args():
    parser = argparse.ArgumentParser(description="一键启动 RAG 后端和前端")
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI 监听地址")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    return parser.parse_args()


def port_is_open(host: str, port: int) -> bool:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((connect_host, port), timeout=0.3):
            return True
    except OSError:
        return False


def wait_for_api(url: str, process: subprocess.Popen, timeout: int = 120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"FastAPI 启动失败，退出码: {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(f"等待 FastAPI 超时: {url}")


def stop_process(process: subprocess.Popen | None):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def main():
    args = parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("未检测到 DEEPSEEK_API_KEY，请先设置环境变量。")
    for label, port in (("FastAPI", args.api_port), ("Streamlit", args.ui_port)):
        if port_is_open(args.host, port):
            raise SystemExit(f"{label} 端口 {port} 已被占用，请先关闭旧服务。")

    api_process = None
    ui_process = None
    env = os.environ.copy()
    env["RAG_API_URL"] = f"http://127.0.0.1:{args.api_port}"
    try:
        print(f"[1/2] 启动 FastAPI: http://127.0.0.1:{args.api_port}")
        api_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                args.host,
                "--port",
                str(args.api_port),
            ],
            cwd=WEB_DIR,
            env=env,
        )
        wait_for_api(f"http://127.0.0.1:{args.api_port}/api/health", api_process)

        print(f"[2/2] 启动 Streamlit: http://localhost:{args.ui_port}")
        ui_command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_ui.py",
            "--server.port",
            str(args.ui_port),
        ]
        if args.no_browser:
            ui_command.extend(["--server.headless", "true"])
        ui_process = subprocess.Popen(ui_command, cwd=WEB_DIR, env=env)

        print("RAG 服务已启动。按 Ctrl+C 可同时关闭前后端。")
        while True:
            api_code = api_process.poll()
            ui_code = ui_process.poll()
            if api_code is not None:
                raise RuntimeError(f"FastAPI 已退出，退出码: {api_code}")
            if ui_code is not None:
                raise RuntimeError(f"Streamlit 已退出，退出码: {ui_code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在关闭 RAG 服务...")
    finally:
        stop_process(ui_process)
        stop_process(api_process)
        print("RAG 服务已关闭。")


if __name__ == "__main__":
    main()

"""阶段 07：启动、查询和恢复长期运行的 ViViT 后台训练任务。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


VIVIT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()
FORECAST_YEARS = tuple(range(2018, 2026))
SEEDS = (42, 43, 44)
EXPECTED_TASKS = len(FORECAST_YEARS) * len(SEEDS)
EXPECTED_PREDICTION_FILES = len(FORECAST_YEARS) - 1
HEARTBEAT_SECONDS = 30.0
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_EPOCH_PATTERN = re.compile(r"epoch=(\d+)")


def _utc_now() -> str:
    """返回带 UTC 时区的当前时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    """读取对应产物，并按接口约定处理缺失文件。"""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def default_screen_session_name(run_name: str) -> str:
    """为一个 run 生成确定、合法且长度受限的 screen 会话名。"""
    cleaned = _SAFE_NAME.sub("_", str(run_name).strip()).strip("._-")
    if not cleaned:
        raise ValueError("run_name 无法生成有效的 screen 会话名")
    value = f"vivit_{cleaned}"
    return value[:80]


def _validate_session_name(session_name: str) -> str:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    value = str(session_name).strip()
    if not value or _SAFE_NAME.search(value):
        raise ValueError(
            "SCREEN_SESSION_NAME 只能包含字母、数字、下划线、连字符和句点"
        )
    return value


def _job_paths(run_name: str, vivit_dir: Path) -> tuple[Path, Path]:
    """返回指定 run 的后台任务清单和总控日志路径。"""
    root = vivit_dir / "data" / "quality" / "runs" / run_name / "background_job"
    return root / "job.json", root / "orchestrator.log"


def _command_signature(command: Sequence[str]) -> str:
    """构造或签名后台任务使用的命令行。"""
    encoded = json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _screen_sessions() -> dict[str, str]:
    """读取 screen 会话并解析其运行状态。"""
    executable = shutil.which("screen")
    if executable is None:
        return {}
    result = subprocess.run(  # 执行命令并收集退出状态
        [executable, "-ls"], text=True, capture_output=True, check=False
    )
    sessions: dict[str, str] = {}
    for line in (result.stdout + result.stderr).splitlines():
        match = re.search(r"\s*(\d+\.\S+)\s+\(", line)
        if match:
            full_name = match.group(1)
            sessions[full_name.split(".", 1)[1]] = full_name
    return sessions


def _training_processes(run_name: str) -> list[dict[str, Any]]:
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    result = subprocess.run(  # 执行命令并收集退出状态
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True
    )
    processes: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if not pid_text.isdigit():
            continue
        if not (
            "stage05_baseline_training.py" in command
            or "stage06_barra_training.py" in command
        ):
            continue
        run_match = re.search(r"--run-name\s+(\S+)", command)
        if run_match is None or run_match.group(1) != run_name:
            continue
        year_match = re.search(r"--worker-year\s+(\d+)", command)
        seed_match = re.search(r"--worker-seed\s+(\d+)", command)
        processes.append(
            {
                "pid": int(pid_text),
                "role": (
                    "worker"
                    if re.search(r"\s(?:worker|ddp-worker)\s", f" {command} ")
                    else "orchestrator"
                ),
                "forecast_year": int(year_match.group(1)) if year_match else None,
                "seed": int(seed_match.group(1)) if seed_match else None,
            }
        )
    return processes


def _last_logged_epoch(path: Path) -> int | None:
    """从 checkpoint 或日志提取训练任务进度。"""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 16384))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = _EPOCH_PATTERN.findall(tail)
    return int(matches[-1]) if matches else None


def _task_progress(run_name: str, vivit_dir: Path) -> list[dict[str, Any]]:
    """从 checkpoint 或日志提取训练任务进度。"""
    model_root = vivit_dir / "models" / "runs" / run_name
    quality_root = vivit_dir / "data" / "quality" / "runs" / run_name
    processes = {
        (item["forecast_year"], item["seed"]): item["pid"]
        for item in _training_processes(run_name)
        if item["role"] == "worker"
    }
    records: list[dict[str, Any]] = []
    for year in FORECAST_YEARS:
        for seed in SEEDS:
            task_root = model_root / f"forecast_{year}" / f"seed_{seed}"
            completed = _read_json(task_root / "completed.json")
            last_epoch = None
            if completed is not None:
                last_epoch = completed.get("summary", {}).get("last_epoch")
                state = "completed"
            else:
                log_name = f"forecast_{year}.seed_{seed}.log"
                last_epoch = _last_logged_epoch(
                    quality_root / "style_task_logs" / log_name
                )
                if last_epoch is None:
                    last_epoch = _last_logged_epoch(
                        quality_root / "a2_task_logs" / log_name
                    )
                if last_epoch is None:
                    last_epoch = _last_logged_epoch(
                        quality_root / "ddp_task_logs" / log_name
                    )
                if (year, seed) in processes:
                    state = "running"
                elif (task_root / "last.pt").exists():
                    state = "resumable"
                else:
                    state = "queued"
            records.append(
                {
                    "forecast_year": year,
                    "seed": seed,
                    "state": state,
                    "last_epoch": last_epoch,
                    "pid": processes.get((year, seed)),
                }
            )
    return records


def training_job_status(
    run_name: str,
    *,
    screen_session_name: str | None = None,
    vivit_dir: Path | str = VIVIT_DIR,
) -> dict[str, Any]:
    """汇总持久化任务状态、screen 存活情况及各年度 checkpoint 进度。"""
    root = Path(vivit_dir).resolve()
    job_path, log_path = _job_paths(run_name, root)
    job = _read_json(job_path) or {}
    session_name = screen_session_name or job.get("screen_session")
    if session_name is None:
        session_name = default_screen_session_name(run_name)
    session_name = _validate_session_name(session_name)
    sessions = _screen_sessions()
    processes = _training_processes(run_name)
    tasks = _task_progress(run_name, root)
    completed_tasks = sum(item["state"] == "completed" for item in tasks)
    resumable_tasks = sum(item["state"] == "resumable" for item in tasks)
    running_tasks = sum(item["state"] == "running" for item in tasks)
    screen_alive = session_name in sessions

    recorded_state = job.get("status")
    prediction_files = len(
        list(
            (
                root / "data" / "predictions" / "runs" / run_name
            ).glob("vivit_predictions.*.parquet")
        )
    )
    if screen_alive or processes:
        state = "running" if job else "running_unmanaged"
    elif recorded_state == "failed":
        state = "failed"
    elif recorded_state == "completed":
        state = (
            "completed"
            if completed_tasks == EXPECTED_TASKS
            and prediction_files == EXPECTED_PREDICTION_FILES
            else "incomplete_outputs"
        )
    elif recorded_state in {"launching", "running"}:
        state = "interrupted"
    elif (
        completed_tasks == EXPECTED_TASKS
        and prediction_files == EXPECTED_PREDICTION_FILES
    ):
        state = "completed_existing"
    elif completed_tasks or resumable_tasks:
        state = "resumable"
    else:
        state = "not_started"

    return {
        "run_name": run_name,
        "status": state,
        "screen_session": session_name,
        "screen_alive": screen_alive,
        "completed_tasks": completed_tasks,
        "expected_tasks": EXPECTED_TASKS,
        "running_tasks": running_tasks,
        "resumable_tasks": resumable_tasks,
        "prediction_files": prediction_files,
        "job_path": str(job_path),
        "log_path": str(log_path),
        "last_heartbeat": job.get("last_heartbeat"),
        "exit_code": job.get("exit_code"),
        "error": job.get("error"),
        "processes": processes,
        "tasks": tasks,
    }


def format_training_job_status(status: dict[str, Any]) -> str:
    """把后台任务状态格式化为 Notebook02 展示的紧凑文本。"""
    lines = [
        f"run_name: {status['run_name']}",
        f"status: {status['status']}",
        f"screen_session: {status['screen_session']}",
        f"screen_alive: {status['screen_alive']}",
        (
            "tasks: "
            f"completed={status['completed_tasks']}/{status['expected_tasks']}, "
            f"running={status['running_tasks']}, "
            f"resumable={status['resumable_tasks']}"
        ),
        f"prediction_files: {status['prediction_files']}",
        f"log: {status['log_path']}",
    ]
    if status.get("last_heartbeat"):
        lines.append(f"last_heartbeat: {status['last_heartbeat']}")
    if status.get("exit_code") is not None:
        lines.append(f"exit_code: {status['exit_code']}")
    if status.get("error"):
        lines.append(f"error: {status['error']}")
    active = [item for item in status["tasks"] if item["state"] == "running"]
    for item in active:
        lines.append(
            "active: "
            f"{item['forecast_year']}/seed={item['seed']} "
            f"epoch={item['last_epoch']} pid={item['pid']}"
        )
    return "\n".join(lines)


def tail_training_job_log(
    run_name: str,
    *,
    lines: int = 30,
    vivit_dir: Path | str = VIVIT_DIR,
) -> str:
    """读取后台总控日志末尾若干行。"""
    if lines < 0:
        raise ValueError("lines 必须 >= 0")
    _, log_path = _job_paths(run_name, Path(vivit_dir).resolve())
    if lines == 0 or not log_path.exists():
        return ""
    return "\n".join(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def launch_training_job(
    command: Sequence[str],
    run_name: str,
    *,
    screen_session_name: str | None = None,
    allow_command_change: bool = False,
    restart_completed: bool = False,
    vivit_dir: Path | str = VIVIT_DIR,
) -> dict[str, Any]:
    """在独立 screen 中启动 A2 总控，或复用已有任务状态。"""
    root = Path(vivit_dir).resolve()
    normalized_command = [str(value) for value in command]
    if not normalized_command:
        raise ValueError("训练命令不能为空")
    if "--run-name" not in normalized_command:
        raise ValueError("训练命令缺少 --run-name")
    index = normalized_command.index("--run-name")  # 仅使用当日截面统计量
    if index + 1 >= len(normalized_command) or normalized_command[index + 1] != run_name:
        raise ValueError("训练命令中的 --run-name 与 run_name 不一致")
    session_name = _validate_session_name(
        screen_session_name or default_screen_session_name(run_name)
    )
    screen_executable = shutil.which("screen")
    if screen_executable is None:
        raise FileNotFoundError("未找到 screen；无法启动持久化后台训练")

    job_path, log_path = _job_paths(run_name, root)
    current_job = _read_json(job_path)
    signature = _command_signature(normalized_command)  # 生成数据复用签名
    sessions = _screen_sessions()
    processes = _training_processes(run_name)
    if session_name in sessions:
        if current_job and current_job.get("command_signature") == signature:
            result = training_job_status(
                run_name, screen_session_name=session_name, vivit_dir=root
            )
            result["launch_action"] = "already_running"
            return result
        raise RuntimeError(
            f"screen 会话 {session_name!r} 已存在，但不属于当前同签名任务"
        )
    if processes:
        raise RuntimeError(
            f"run {run_name!r} 已有训练进程，但不在受管 screen 会话中；"
            "为避免重复占用 GPU，本次不启动"
        )
    if current_job and current_job.get("command_signature") != signature:
        if not allow_command_change:
            raise ValueError(
                "同名后台任务的训练命令签名已变化；请使用新 run name，"
                "或显式设置 OVERWRITE=1"
            )
    existing = training_job_status(
        run_name, screen_session_name=session_name, vivit_dir=root
    )
    if (
        existing["completed_tasks"] == EXPECTED_TASKS
        and existing["prediction_files"] == EXPECTED_PREDICTION_FILES
        and not restart_completed
    ):
        existing["launch_action"] = "reuse_completed"
        return existing

    attempt = int((current_job or {}).get("attempt", 0)) + 1
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "screen_session": session_name,
        "status": "launching",
        "command": normalized_command,
        "command_signature": signature,
        "cwd": str(root),
        "job_path": str(job_path),
        "log_path": str(log_path),
        "attempt": attempt,
        "created_at": (current_job or {}).get("created_at", _utc_now()),
        "launched_at": _utc_now(),
        "last_heartbeat": None,
        "exit_code": None,
        "error": None,
    }
    _atomic_write_json(job_path, payload)  # 原子落盘，避免半成品
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(  # 执行命令并收集退出状态
            [
                screen_executable,
                "-dmS",
                session_name,
                "-L",
                "-Logfile",
                str(log_path),
                sys.executable,
                str(SCRIPT_PATH),
                "run-job",
                "--job-file",
                str(job_path),
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except BaseException as error:
        payload.update(
            {
                "status": "failed",
                "error": f"screen launch failed: {type(error).__name__}: {error}",
                "finished_at": _utc_now(),
            }
        )
        _atomic_write_json(job_path, payload)  # 原子落盘，避免半成品
        raise
    time.sleep(0.2)
    result = training_job_status(
        run_name, screen_session_name=session_name, vivit_dir=root
    )
    result["launch_action"] = "launched"
    return result


def run_job(job_file: Path | str) -> int:
    """在 screen 内执行持久化任务清单，并记录心跳、退出码和异常。"""
    path = Path(job_file).resolve()
    payload = _read_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    payload.update(
        {
            "status": "running",
            "runner_pid": os.getpid(),
            "started_at": _utc_now(),
            "last_heartbeat": _utc_now(),
            "exit_code": None,
            "error": None,
        }
    )
    _atomic_write_json(path, payload)  # 原子落盘，避免半成品
    process: subprocess.Popen[str] | None = None
    try:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(  # 启动独立训练任务
            payload["command"],
            cwd=payload["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        payload["orchestrator_pid"] = process.pid
        _atomic_write_json(path, payload)  # 原子落盘，避免半成品
        while True:
            try:
                code = process.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                payload["last_heartbeat"] = _utc_now()
                _atomic_write_json(path, payload)  # 原子落盘，避免半成品
        payload.update(
            {
                "status": "completed" if code == 0 else "failed",
                "exit_code": int(code),
                "finished_at": _utc_now(),
                "last_heartbeat": _utc_now(),
            }
        )
        _atomic_write_json(path, payload)  # 原子落盘，避免半成品
        return int(code)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
        payload.update(
            {
                "status": "failed",
                "exit_code": None,
                "error": f"{type(error).__name__}: {error}",
                "finished_at": _utc_now(),
                "last_heartbeat": _utc_now(),
            }
        )
        _atomic_write_json(path, payload)  # 原子落盘，避免半成品
        raise


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runner = subparsers.add_parser("run-job")
    runner.add_argument("--job-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """根据命令行子命令分派数据、训练、验证或评价流程。"""
    args = parse_args()
    if args.command == "run-job":
        return run_job(args.job_file)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

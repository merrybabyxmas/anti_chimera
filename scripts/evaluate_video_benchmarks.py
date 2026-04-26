from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def _run_command(cmd: List[str], cwd: Optional[str] = None) -> Dict[str, object]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return {
        'cmd': cmd,
        'cwd': cwd,
        'returncode': proc.returncode,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
    }


def _maybe_find_repo_default(candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def evaluate_vbench(videos_path: str, dimension: str, mode: str, repo_root: Optional[str] = None) -> Dict[str, object]:
    videos_path = str(Path(videos_path).resolve())
    if shutil.which('vbench'):
        cmd = ['vbench', 'evaluate', '--dimension', dimension, '--videos_path', videos_path, '--mode', mode]
        return {'backend': 'vbench', 'available': True, 'result': _run_command(cmd)}

    repo_root = repo_root or _maybe_find_repo_default([
        './VBench',
        './deps/VBench',
        '../VBench',
        '../deps/VBench',
    ])
    if repo_root:
        evaluate_py = Path(repo_root) / 'evaluate.py'
        if evaluate_py.exists():
            cmd = ["python", "evaluate.py", "--dimension", dimension, "--videos_path", videos_path, "--mode", mode]
            return {'backend': 'vbench', 'available': True, 'result': _run_command(cmd, cwd=repo_root)}

    return {'backend': 'vbench', 'available': False, 'reason': 'vbench CLI or repo not found'}


def evaluate_videobench(videos_path: str, dimension: str, config_path: Optional[str], models: List[str], repo_root: Optional[str] = None) -> Dict[str, object]:
    videos_path = str(Path(videos_path).resolve())
    if config_path is not None:
        config_path = str(Path(config_path).resolve())
    if shutil.which('videobench'):
        cmd = ['videobench', '--dimension', dimension, '--videos_path', videos_path]
        if config_path:
            cmd += ['--config_path', config_path]
        if models:
            cmd += ['--models', *models]
        return {'backend': 'videobench', 'available': True, 'result': _run_command(cmd)}

    repo_root = repo_root or _maybe_find_repo_default([
        './Video-Bench',
        './deps/Video-Bench',
        '../Video-Bench',
        '../deps/Video-Bench',
    ])
    if repo_root:
        evaluate_py = Path(repo_root) / 'evaluate.py'
        if evaluate_py.exists():
            cmd = ["python", "evaluate.py", "--dimension", dimension, "--videos_path", videos_path]
            if config_path:
                cmd += ['--config_path', config_path]
            if models:
                cmd += ['--models', *models]
            return {'backend': 'videobench', 'available': True, 'result': _run_command(cmd, cwd=repo_root)}

    return {'backend': 'videobench', 'available': False, 'reason': 'videobench CLI or repo not found'}


def main() -> None:
    parser = argparse.ArgumentParser(description='Run VBench and/or Video-Bench evaluation wrappers.')
    parser.add_argument('--videos-path', required=True)
    parser.add_argument('--backend', choices=['vbench', 'videobench', 'both'], default='both')
    parser.add_argument('--dimension', default='subject_consistency')
    parser.add_argument("--videobench-dimension", default="temporal_consistency")
    parser.add_argument('--mode', default='custom_input')
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--models', nargs='*', default=[])
    parser.add_argument('--vbench-repo', default=None)
    parser.add_argument('--videobench-repo', default=None)
    parser.add_argument('--output-report', default=None)
    args = parser.parse_args()

    report: Dict[str, object] = {
        'videos_path': args.videos_path,
        'backend': args.backend,
        'dimension': args.dimension,
        "videobench_dimension": args.videobench_dimension,
        'mode': args.mode,
        'config_path': args.config_path,
        'models': args.models,
    }


    if args.backend in {"vbench", "both"}:
        report["vbench"] = evaluate_vbench(args.videos_path, args.dimension, args.mode, args.vbench_repo)
    if args.backend in {"videobench", "both"}:
        report["videobench"] = evaluate_videobench(args.videos_path, args.videobench_dimension, args.config_path, args.models, args.videobench_repo)
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    print(report_text)
    if args.output_report:
        Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_report).write_text(report_text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

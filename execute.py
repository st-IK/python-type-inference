"""
Test launcher for main.py.

CLI:
    python execute.py <input_python_file> <output_directory>

GUI:
    python execute.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def select_paths_with_tkinter() -> tuple[str, str] | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    try:
        input_file = filedialog.askopenfilename(
            title="解析するPythonファイルを選択",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if not input_file:
            return None

        output_dir = filedialog.askdirectory(
            title="出力フォルダを選択",
        )
        if not output_dir:
            return None

        return input_file, output_dir
    finally:
        root.destroy()


def run_parser(input_file: str, output_dir: str) -> int:
    main_py = Path(__file__).resolve().parent / "main.py"

    if not main_py.exists():
        print(f"main.py was not found: {main_py}", file=sys.stderr)
        return 1

    command = [sys.executable, str(main_py), input_file, output_dir]

    print("[Executor]")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))
    print()

    completed = subprocess.run(command)
    return completed.returncode


def main() -> int:
    if len(sys.argv) == 3:
        return run_parser(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 1:
        selected = select_paths_with_tkinter()
        if selected is None:
            print("Cancelled.")
            return 0
        return run_parser(*selected)

    print(
        "Usage:\n"
        "  python execute.py <input_python_file> <output_directory>\n"
        "  python execute.py",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

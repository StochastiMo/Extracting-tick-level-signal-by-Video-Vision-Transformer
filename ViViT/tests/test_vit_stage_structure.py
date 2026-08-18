from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


VIT_DIR = Path(__file__).resolve().parents[2] / "ViT"
STAGE_FILES = (
    "vit_stage01_foundation.py",
    "vit_stage02_universe.py",
    "vit_stage03_labels.py",
    "vit_stage04_feature_store.py",
    "vit_stage05_model.py",
    "vit_stage06_ranking.py",
    "vit_stage07_training.py",
    "vit_stage08_factor_evaluation.py",
)
LEGACY_FILES = ("vit_data.py", "vit_train.py", "vit_pipeline.py")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
STAGE_PATTERN = re.compile(r"^vit_stage(\d{2})_")


class ViTStageLayoutTests(unittest.TestCase):
    def test_only_ordered_stage_modules_remain(self) -> None:
        self.assertEqual(
            tuple(sorted(path.name for path in VIT_DIR.glob("*.py"))),
            STAGE_FILES,
        )
        for filename in LEGACY_FILES:
            self.assertFalse((VIT_DIR / filename).exists(), filename)
        self.assertFalse((VIT_DIR / "vit_core").exists())

    def test_dependencies_only_point_to_earlier_stages(self) -> None:
        for filename in STAGE_FILES:
            current = int(STAGE_PATTERN.match(filename).group(1))
            tree = ast.parse((VIT_DIR / filename).read_text(encoding="utf-8"))
            dependencies: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    dependencies.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    dependencies.add(node.module)
            for dependency in dependencies:
                match = STAGE_PATTERN.match(dependency)
                if match is not None:
                    self.assertLess(
                        int(match.group(1)), current,
                        f"{filename} 不应反向依赖 {dependency}",
                    )

    def test_all_function_signatures_and_docs_expose_formats(self) -> None:
        failures: list[str] = []
        for filename in STAGE_FILES:
            tree = ast.parse((VIT_DIR / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                arguments = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                if node.args.vararg is not None:
                    arguments.append(node.args.vararg)
                if node.args.kwarg is not None:
                    arguments.append(node.args.kwarg)
                missing = [
                    argument.arg
                    for argument in arguments
                    if argument.arg not in {"self", "cls"}
                    and argument.annotation is None
                ]
                doc = ast.get_docstring(node) or ""
                if (
                    missing
                    or node.returns is None
                    or "输入：" not in doc
                    or "输出：" not in doc
                    or CHINESE_PATTERN.search(doc) is None
                ):
                    failures.append(f"{filename}:{node.lineno}:{node.name}")
        self.assertEqual(failures, [])


    def test_each_logic_statement_has_adjacent_chinese_comment(self) -> None:
        """每条可执行语句及控制分支都必须由紧邻的中文注释说明。"""
        branch_pattern = re.compile(
            r"^(else\s*:|except\b.*:|finally\s*:|case\b.*:)\s*(?:#.*)?$"
        )
        failures: list[str] = []
        for filename in STAGE_FILES:
            source = (VIT_DIR / filename).read_text(encoding="utf-8")
            source_lines = source.splitlines()
            tree = ast.parse(source)
            anchors: set[int] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.stmt):
                    continue
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    continue
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    anchor = min(
                        (item.lineno for item in node.decorator_list),
                        default=node.lineno,
                    )
                else:
                    anchor = node.lineno
                anchors.add(anchor)
            anchors.update(
                lineno
                for lineno, line in enumerate(source_lines, start=1)
                if branch_pattern.match(line.strip())
            )
            for lineno in sorted(anchors):
                previous = lineno - 2
                while previous >= 0 and not source_lines[previous].strip():
                    previous -= 1
                covered = (
                    previous >= 0
                    and source_lines[previous].lstrip().startswith("#")
                    and CHINESE_PATTERN.search(source_lines[previous]) is not None
                )
                if not covered:
                    failures.append(f"{filename}:{lineno}")
        self.assertEqual(failures, [])
    def test_audit_functions_follow_main_workflows(self) -> None:
        expected = {
            "vit_stage02_universe.py": ("build_universe", {"_daily_quality_record"}),
            "vit_stage07_training.py": (
                "run_preflight",
                {
                    "validate_training_inputs", "loader_batch_summary",
                    "validate_rolling_split", "configured_batch_summaries",
                    "validate_outputs", "show_status", "validate_a2_args",
                },
            ),
        }
        for filename, (boundary, audit_names) in expected.items():
            tree = ast.parse((VIT_DIR / filename).read_text(encoding="utf-8"))
            positions = {
                node.name: node.lineno
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for name in audit_names:
                self.assertGreater(positions[name], positions[boundary])


if __name__ == "__main__":
    unittest.main()

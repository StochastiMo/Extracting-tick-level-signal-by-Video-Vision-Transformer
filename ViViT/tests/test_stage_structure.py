from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


VIVIT_DIR = Path(__file__).resolve().parents[1]
STAGE_FILES = (
    "stage01_sequence_data.py",
    "stage02_market_data.py",
    "stage02_barra_data.py",
    "stage02_barra_algebra.py",
    "stage03_model.py",
    "stage04_experiments.py",
    "stage04_no0_baseline.py",
    "stage04_no1_neutral.py",
    "stage04_no2_r2_penalty.py",
    "stage04_no3_adversarial.py",
    "stage04_no4_gate.py",
    "stage04_no5_strict_neutral.py",
    "stage04_no6_dual_tower.py",
    "stage05_baseline_training.py",
    "stage06_barra_training.py",
    "stage07_training_jobs.py",
    "stage08_factor_evaluation.py",
    "stage09_barra_postprocess.py",
)
LEGACY_FILES = (
    "vivit_pipeline.py",
    "vivit_style_neutral_data.py",
    "vivit_style_neutral_training.py",
    "vivit_training_jobs.py",
    "vivit_factor_eval.py",
    "vivit_barra_neutral.py",
)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


class StageLayoutTests(unittest.TestCase):
    def test_stage_modules_replace_legacy_layout(self) -> None:
        for filename in STAGE_FILES:
            self.assertTrue((VIVIT_DIR / filename).is_file(), filename)
        for filename in LEGACY_FILES:
            self.assertFalse((VIVIT_DIR / filename).exists(), filename)
        self.assertFalse((VIVIT_DIR / "vivit_core").exists())

    def test_all_function_signatures_and_docs_expose_formats(self) -> None:
        failures: list[str] = []
        for filename in STAGE_FILES:
            tree = ast.parse(
                (VIVIT_DIR / filename).read_text(encoding="utf-8")
            )
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
                missing_arguments = [
                    argument.arg
                    for argument in arguments
                    if argument.arg not in {"self", "cls"}
                    and argument.annotation is None
                ]
                doc = ast.get_docstring(node) or ""
                if (
                    missing_arguments
                    or node.returns is None
                    or "输入：" not in doc
                    or "输出：" not in doc
                    or CHINESE_PATTERN.search(doc) is None
                ):
                    failures.append(
                        f"{filename}:{node.lineno}:{node.name}"
                    )
        self.assertEqual(failures, [])


    def test_each_logic_statement_has_adjacent_chinese_comment(self) -> None:
        """每条可执行语句及控制分支都必须由紧邻的中文注释说明。"""
        branch_pattern = re.compile(
            r"^(else\s*:|except\b.*:|finally\s*:|case\b.*:)\s*(?:#.*)?$"
        )
        failures: list[str] = []
        for filename in STAGE_FILES:
            source = (VIVIT_DIR / filename).read_text(encoding="utf-8")
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
    def test_audit_sections_follow_main_workflows(self) -> None:
        boundaries = {
            "stage01_sequence_data.py": "set_train_epoch",
            "stage05_baseline_training.py": "run_memory_benchmark",
            "stage06_barra_training.py": "finalize_run",
            "stage08_factor_evaluation.py": "compare_prediction_signal_pearson",
            "stage09_barra_postprocess.py": "build_barra_residual_run",
        }
        audit_names = {
            "stage01_sequence_data.py": {
                "validate_history_indices", "audit_split_alignment",
                "loader_batch_summary", "require_source_data",
                "require_aligned_sequence_store",
            },
            "stage05_baseline_training.py": {
                "validate_training_config", "validate_outputs",
            },
            "stage06_barra_training.py": {
                "measure_daily_score_dispersion", "measure_style_correlations",
                "summarize_adversarial_predictions", "run_adversarial_probes",
                "validate_outputs", "write_report", "smoke_test",
            },
            "stage08_factor_evaluation.py": {
                "audit_evaluation_inputs", "_validated_evaluation_manifest",
            },
            "stage09_barra_postprocess.py": {"audit_barra_inputs"},
        }
        for filename, boundary_name in boundaries.items():
            tree = ast.parse(
                (VIVIT_DIR / filename).read_text(encoding="utf-8")
            )
            positions = {
                node.name: node.lineno
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for audit_name in audit_names[filename]:
                self.assertGreater(
                    positions[audit_name], positions[boundary_name],
                    f"{filename}:{audit_name} 应位于主流程之后",
                )


if __name__ == "__main__":
    unittest.main()

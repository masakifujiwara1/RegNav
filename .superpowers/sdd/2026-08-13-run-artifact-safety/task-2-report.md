# Task 2 実施報告

## RED

`tests/test_preserve_run_artifacts.py` を追加し、実装前に `scripts.preserve_run_artifacts` が存在しないため focused test が `ModuleNotFoundError` で失敗することを確認した。

## GREEN

`preserve_artifacts(sources, destination)` を stdlib-only で追加した。全sourceを事前検証し、既存destinationは即時拒否、`TemporaryDirectory` に `copy2` で退避、コピー後の内容で SHA256 を計算し、`SHA256SUMS.json` を書いてから `os.replace` でdestinationへ原子的に切り替える。CLIは brief 指定の5 artifact引数と `--destination` を受け付ける。

## テスト

- RED: `uv run pytest tests/test_preserve_run_artifacts.py -v` -> `ModuleNotFoundError: No module named 'scripts'`
- GREEN: `uv run pytest tests/test_preserve_run_artifacts.py -v` -> 4 passed
- FULL: `uv run pytest` -> 52 passed
- `git diff --check` -> passed

## 自己レビュー

- source は copy 前に全件 `is_file()` で検証し、欠損時は partial copy を残さない。
- destination 既存時は temp dir 作成前に `FileExistsError`。
- hash は source ではなく copied artifact から算出し、保存内容と manifest を一致させる。
- manifest は `sort_keys=True` で安定化。
- `scripts/` は pytest import path に自動追加されないため、テストは script ファイルを直接 load して CLI/関数の実挙動を検証した。

## 懸念

source key の相対パス妥当性（`..` など）は brief 範囲外のため未検証。

## Fix round 1

- review finding: `sources` key を無検証で `staged / relative_name` に渡しており、absolute path や `..` で staging 外へ書ける。
- RED: `test_preserve_artifacts_rejects_absolute_and_parent_escape_keys` を追加し、`uv run pytest tests/test_preserve_run_artifacts.py -v` で `Failed: DID NOT RAISE ValueError` を確認した。
- GREEN: `_validate_relative_name()` を追加し、absolute path と `..` segment を `ValueError` で拒否するようにした。CLI の固定 key は相対 path のまま維持。
- VERIFY: `uv run pytest tests/test_preserve_run_artifacts.py -v` -> 5 passed、`uv run pytest` -> 53 passed、`git diff --check` -> passed。
- note: テストで absolute key と `../escape.pt` key を明示的に拒否し、destination 外ファイル未生成も確認した。


## Fix round 1/5 (final review TOCTOU)

- review finding: `destination.exists()` と `os.replace()` の間で destination が作られると上書きし得た。
- RED: `test_preserve_artifacts_does_not_clobber_destination_created_after_precheck` を追加し、出力先が precheck 後に出現する race を再現した。
- GREEN: Linux では `renameat2(RENAME_NOREPLACE)` を `ctypes` 経由で使う no-clobber publish に変更した。非対応環境では atomicity より no-clobber を優先し、`mkdir()` 後に staged children を移す fallback とした。
- VERIFY: `uv run pytest tests/test_preserve_run_artifacts.py tests/test_build_comparison_report.py` -> 18 passed、`uv run pytest` -> 66 passed、`git diff --check` -> passed。
- note: fallback は fully atomic ではないが、review finding の「existing contents を壊さない」を優先している。

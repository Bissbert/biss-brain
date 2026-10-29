"""Parity: recall CLI, MCP search_memory and UserPromptSubmit inject all rank
through scripts/retrieve.py, so on the same fixture they must agree with each
other and with the recorded golden order.

The golden pins the ranking so an unintended change shows up as a diff.
After an intended ranking change, regenerate it and review the diff:

    python tests/test_retrieve_parity.py --regen
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import _retrieval_fixture as fx  # noqa: E402

GOLDEN_PATH = TESTS / "fixtures" / "retrieve_parity_golden.json"
MODES = ["recall", "mcp", "inject"]
# inject prints "variant N" from the text; map N back to the chunk id prefix
VARIANT_ID = {str(i): hashlib.sha256(f"chunk-{i}".encode()).hexdigest()[:8] for i in range(30)}


def run_mode(root: Path, scenario: tuple[str, bool, bool], mode: str) -> list:
    _, hybrid, summary_first = scenario
    root.mkdir(parents=True, exist_ok=True)
    env = fx.env_for(root, hybrid, summary_first)
    p = subprocess.run(
        [sys.executable, str(TESTS / "_retrieval_driver.py"), mode, str(root)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.split("@@JSON@@", 1)[1])


@pytest.mark.parametrize("scenario", fx.scenarios(), ids=lambda s: s[0])
@pytest.mark.parametrize("mode", MODES)
def test_read_path_matches_golden(tmp_path, scenario, mode):
    golden = json.loads(GOLDEN_PATH.read_text())
    assert run_mode(tmp_path, scenario, mode) == golden[f"{scenario[0]}/{mode}"]


@pytest.mark.parametrize("scenario", fx.scenarios(), ids=lambda s: s[0])
def test_read_paths_agree(tmp_path, scenario):
    """MCP equals recall; inject (top 3, gate open) is the head of the same order.

    Inject ignores prefer_summaries_in_recall by design (it picks by prompt
    relevance alone), so with summary_first it is compared to nothing.
    """
    recall = run_mode(tmp_path / "r", scenario, "recall")
    mcp = run_mode(tmp_path / "m", scenario, "mcp")
    assert mcp == recall
    if scenario[2]:
        return
    inject = run_mode(tmp_path / "i", scenario, "inject")
    for q, ids in recall.items():
        assert [VARIANT_ID[v] for v in inject[q]] == ids[:3], q


def regen(root: Path) -> None:
    out = {}
    for sc in fx.scenarios():
        for mode in MODES:
            d = root / f"{sc[0]}-{mode}"
            out[f"{sc[0]}/{mode}"] = run_mode(d, sc, mode)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__" and "--regen" in sys.argv:
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        regen(Path(t))

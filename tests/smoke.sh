#!/usr/bin/env bash
# End-to-end smoke test for biss-brain.
# Uses passthrough distill + BISS_BRAIN_FAKE_EMBED=1 so no claude/Ollama calls required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SCRIPTS="$ROOT/scripts"

# ── helpers ─────────────────────────────────────────────────────────────
pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAILURES=$((FAILURES + 1)); }
FAILURES=0

# macOS-compatible mktemp
_mktemp_file() { mktemp "${TMPDIR:-/tmp}/biss-brain-test.XXXXXX"; }
_mktemp_dir()  { mktemp -d "${TMPDIR:-/tmp}/biss-brain-test.XXXXXX"; }

# ── test workspace ───────────────────────────────────────────────────────
TEST_CWD="$(_mktemp_dir)"
TRANSCRIPT_PATH="$(_mktemp_file)"
TEST_CONFIG="$(_mktemp_file)"
FLUSH_LOG="$(_mktemp_file)"

# Write test config: passthrough distill, short debounce (3s), fake embed dim
cat > "$TEST_CONFIG" << 'TOML'
[distill]
backend = "passthrough"

[debounce]
stop_seconds = 3
subagent_stop_seconds = 3
max_events_per_batch = 50
segment_idle_seconds = 3
idle_exit_seconds = 15

[embed]
backend = "ollama"
model = "nomic-embed-text"
dim = 768
endpoint = "http://127.0.0.1:11434"

[retrieve]
hybrid = false
TOML

export BISS_BRAIN_CONFIG="$TEST_CONFIG"
export BISS_BRAIN_FAKE_EMBED=1
export CLAUDE_PLUGIN_ROOT="$ROOT"

# ── derive project hash ──────────────────────────────────────────────────
PROJ_KEY=$("$PYTHON" -c "
import sys; sys.path.insert(0, '$SCRIPTS')
from common import project_key
print(project_key('$TEST_CWD'))
")
BRAIN_ROOT="$HOME/.claude/brain"
PROJ_DIR="$BRAIN_ROOT/$PROJ_KEY"

echo "smoke test"
echo "  TEST_CWD:    $TEST_CWD"
echo "  PROJ_KEY:    $PROJ_KEY"
echo "  PROJ_DIR:    $PROJ_DIR"
echo ""

# ── Step 1: Clean slate ──────────────────────────────────────────────────
echo "Step 1: wipe $PROJ_DIR"
rm -rf "$PROJ_DIR"

# ── Step 2: Create fake transcript ──────────────────────────────────────
echo "Step 2: create fake transcript at $TRANSCRIPT_PATH"
SESSION_ID="smoketest-session-abc123"
cat > "$TRANSCRIPT_PATH" << 'JSONL'
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"I decided to use LanceDB for the vector store because it natively supports hybrid BM25+vector search."}]},"session_id":"smoketest-session-abc123"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"The SQLite metadata table tracks superseded_by as empty string sentinel to avoid NULL comparisons."}]},"session_id":"smoketest-session-abc123"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Embedding model nomic-embed-text produces 768-dimensional vectors suitable for semantic search."}]},"session_id":"smoketest-session-abc123"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"The debounce window collapses rapid Stop events from the same session into a single distillation call."}]},"session_id":"smoketest-session-abc123"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Bootstrap script pulls the embedding model and copies config.toml.example to ~/.claude/brain/config.toml."}]},"session_id":"smoketest-session-abc123"}
JSONL

# ── Step 3: Feed capture.py ──────────────────────────────────────────────
echo "Step 3: fire capture.py stop"
HOOK_PAYLOAD="{\"session_id\":\"$SESSION_ID\",\"cwd\":\"$TEST_CWD\",\"transcript_path\":\"$TRANSCRIPT_PATH\"}"
echo "$HOOK_PAYLOAD" | PYTHONPATH="$SCRIPTS" "$PYTHON" "$SCRIPTS/capture.py" stop

# ── Step 4: Start flush daemon and wait for it to process ────────────────
echo "Step 4: start flush daemon and wait for debounce + processing"

# Kill any stale flush daemon that might be locked
FLUSH_PID_FILE="$BRAIN_ROOT/flush.pid"
if [ -f "$FLUSH_PID_FILE" ]; then
    STALE_PID=$(cat "$FLUSH_PID_FILE" 2>/dev/null || true)
    if [ -n "$STALE_PID" ]; then
        kill "$STALE_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$FLUSH_PID_FILE"
fi

# Spawn flush.py directly (not via _run.sh so we keep env vars)
PYTHONPATH="$SCRIPTS" "$PYTHON" "$SCRIPTS/flush.py" > "$FLUSH_LOG" 2>&1 &
FLUSH_PID=$!
echo "  flush daemon PID: $FLUSH_PID"

# Wait: debounce_seconds(3) + processing time. Max 30s total.
WAIT=0
MAX_WAIT=30
INDEXED=0
while [ $WAIT -lt $MAX_WAIT ]; do
    sleep 2
    WAIT=$((WAIT + 2))
    # Check if SQLite has a row
    ROW_COUNT=$("$PYTHON" -c "
import sqlite3, sys, os
sys.path.insert(0, '$SCRIPTS')
from common import meta_db_path
db_path = meta_db_path('$TEST_CWD')
if not os.path.exists(str(db_path)):
    print(0)
    sys.exit(0)
conn = sqlite3.connect(str(db_path))
count = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
print(count)
conn.close()
" 2>/dev/null || echo 0)
    if [ "$ROW_COUNT" -ge 1 ] 2>/dev/null; then
        echo "  indexed $ROW_COUNT chunk(s) after ${WAIT}s"
        INDEXED=1
        break
    fi
    echo "  waiting... ${WAIT}s elapsed (chunks so far: $ROW_COUNT)"
done

# Kill flush daemon
kill "$FLUSH_PID" 2>/dev/null || true
wait "$FLUSH_PID" 2>/dev/null || true

if [ "$INDEXED" -eq 0 ]; then
    fail "flush daemon did not index any chunks within ${MAX_WAIT}s"
    echo "--- flush.log ---"
    cat "$FLUSH_LOG" || true
    echo "--- end flush.log ---"
else
    pass "flush daemon indexed chunks"
fi

# ── Step 5: recall.py query ──────────────────────────────────────────────
echo ""
echo "Step 5: query recall.py"
RECALL_OUT=$(cd "$TEST_CWD" && PYTHONPATH="$SCRIPTS" "$PYTHON" "$SCRIPTS/recall.py" "LanceDB hybrid search vector store" 2>&1 || true)
echo "$RECALL_OUT"

if echo "$RECALL_OUT" | grep -q "no memory yet\|no matches"; then
    fail "recall.py returned empty result"
elif [ -z "$RECALL_OUT" ]; then
    fail "recall.py produced no output"
else
    pass "recall.py returned non-empty results"
fi

# ── Step 6: Assertions ───────────────────────────────────────────────────
echo ""
echo "Step 6: assert SQLite has >= 1 row in chunks"
FINAL_COUNT=$("$PYTHON" -c "
import sqlite3, sys, os
sys.path.insert(0, '$SCRIPTS')
from common import meta_db_path
db_path = meta_db_path('$TEST_CWD')
if not os.path.exists(str(db_path)):
    print(0)
    sys.exit(0)
conn = sqlite3.connect(str(db_path))
count = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
print(count)
conn.close()
" 2>/dev/null || echo 0)
if [ "$FINAL_COUNT" -ge 1 ] 2>/dev/null; then
    pass "meta.sqlite has $FINAL_COUNT chunk(s)"
else
    fail "meta.sqlite has 0 chunks — expected >= 1"
fi

# ── Step 7: UI server endpoints ──────────────────────────────────────────
echo ""
echo "Step 7: spawn UI server and curl all panel endpoints"

# Pick a free port
UI_PORT=$("$PYTHON" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
UI_LOG="$(_mktemp_file)"

cd "$ROOT" && PYTHONPATH="$ROOT:$SCRIPTS" "$PYTHON" -m uvicorn ui.server:app \
    --host 127.0.0.1 --port "$UI_PORT" --log-level warning > "$UI_LOG" 2>&1 &
UI_PID=$!
echo "  UI server PID: $UI_PID on port $UI_PORT"

# Wait for /healthz
WAIT=0
UI_READY=0
while [ $WAIT -lt 10 ]; do
    if curl -sf "http://127.0.0.1:$UI_PORT/healthz" > /dev/null 2>&1; then
        UI_READY=1
        break
    fi
    sleep 0.5
    WAIT=$((WAIT + 1))
done

if [ "$UI_READY" -eq 0 ]; then
    fail "UI server failed to start within 5s"
    echo "--- ui.log ---"; cat "$UI_LOG" || true; echo "--- end ui.log ---"
else
    pass "UI server /healthz responds"

    # Hit each panel endpoint
    BASE="http://127.0.0.1:$UI_PORT"
    for ep in "/" \
              "/search?project=$PROJ_KEY&q=lancedb" \
              "/sessions?project=$PROJ_KEY" \
              "/api/sessions/recent?project=$PROJ_KEY" \
              "/timeline?project=$PROJ_KEY" \
              "/api/timeline.json?project=$PROJ_KEY" \
              "/scatter?project=$PROJ_KEY" \
              "/api/scatter.json?project=$PROJ_KEY" \
              "/graph?project=$PROJ_KEY" \
              "/api/graph.json?project=$PROJ_KEY"; do
        STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$BASE$ep" || echo "000")
        if [ "$STATUS" = "200" ]; then
            pass "GET $ep → 200"
        else
            fail "GET $ep → $STATUS"
        fi
    done
fi

kill "$UI_PID" 2>/dev/null || true
wait "$UI_PID" 2>/dev/null || true
rm -f "$UI_LOG"

# ── Step 8: Cleanup ──────────────────────────────────────────────────────
echo ""
echo "Step 8: cleanup"
rm -rf "$PROJ_DIR" "$TEST_CWD" "$TEST_CONFIG" "$FLUSH_LOG"
rm -f "$TRANSCRIPT_PATH"
pass "cleanup complete"

# ── Result ───────────────────────────────────────────────────────────────
echo ""
if [ "$FAILURES" -eq 0 ]; then
    echo "smoke test PASSED (all checks green)"
    exit 0
else
    echo "smoke test FAILED ($FAILURES failure(s))"
    exit 1
fi

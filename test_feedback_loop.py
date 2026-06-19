"""End-to-end check for the feedback loop: an entry added via the daily check-in path
must be retrievable through rag.search() — otherwise next week's /plan can't learn from it.

Run with:  .venv/bin/python test_feedback_loop.py

Note: this adds one clearly-marked test entry to the live RAG index (and feedback log);
the unique TESTMARKER makes it easy to spot/ignore.
"""
from datetime import date

from src import rag, scheduler

MARKER = "TESTMARKER42"
INSIGHT = f"Plattfuß-Übungen abends nie geschafft, lieber morgens einplanen {MARKER}"


def main() -> None:
    before = rag.get_doc_count()
    print(f"RAG documents before: {before}")

    # Exercise the real append path used by the evening/weekly check-in.
    scheduler._append_feedback(INSIGHT, weekly=False)

    after = rag.get_doc_count()
    print(f"RAG documents after:  {after}")

    # The feedback line must show up in a semantic search a future plan would run.
    result = rag.search("Wann soll ich die Plattfuß-Übungen machen?", k=5)
    assert MARKER in result, (
        "FAIL: feedback entry not retrievable via rag.search().\n"
        f"Search returned:\n{result}"
    )

    # And it must be persisted to the feedback log (survives a full rebuild on restart).
    with open(scheduler.FEEDBACK_PATH) as f:
        log_contents = f.read()
    stamp = date.today().isoformat()
    assert f"[{stamp}]" in log_contents and MARKER in log_contents, (
        "FAIL: feedback entry not written to feedback_history.txt"
    )

    print("\n✅ PASS: feedback entry is retrievable via RAG search and persisted to the log.")
    print("Search snippet:\n" + result[:300])


if __name__ == "__main__":
    main()

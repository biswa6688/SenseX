"""Short-span speaker resolution via multiple weak signals.

Voice embeddings are unreliable for sub-1.5s / 1-4-word utterances — too
little acoustic signal for confident classification on their own.
Verified on real audio (ground truth from an independent transcript): a
diarization-stage misclustering of a single confirming word ("Perfect.")
couldn't be caught by either the gap-word fix (merge.py's
_classify_gap) or turn-level boundary refinement (boundary_refinement.py)
— it's covered by a real (but wrong) raw diarization turn, not a gap,
and far too short to be a useful window-scan candidate on its own.

Operates at the WORD level, after merge.py's initial speaker assignment
but before words are grouped into turns — this is what lets it catch a
short backchannel-shaped word run trailing off the end of an otherwise-
correct longer turn (the "...Perfect." case), not just already-isolated
short turns.

Combines four weaker signals into one score for each candidate short,
same-speaker word run that's immediately sandwiched between two runs of
one OTHER speaker (the classic "everyone around it says X, this one
short run differs" pattern):

  linguistic — does the already-loaded LLM judge the text as a brief
               acknowledgment/backchannel response, as opposed to a
               substantive continuation? A real language-model judgment,
               not a hardcoded word list — generalizes across languages
               and phrasing a fixed vocabulary never would (see
               _linguistic_score, DECISIONS.md #19).
  position   — is it sandwiched by the SAME other speaker on both sides?
               (required precondition, not just a scored signal here.)
  pause      — how much silence precedes the run (a real pause favors a
               genuine turn-taking event over a same-speaker disfluency)?
  acoustic   — embedding similarity to each neighboring speaker, as
               supporting evidence only (weighted lowest — see above).

Reassigns only when the combined score clears REASSIGN_THRESHOLD — never
flips a run on any single signal alone, and never hardcodes "perfect" or
any other individual word as a special case.
"""

import logging

import numpy as np

from app.pipeline.embedding_utils import embed_span
from app.pipeline.llm_analysis import get_llm
from app.pipeline.stt import Word

logger = logging.getLogger(__name__)

MAX_SHORT_RUN_SECONDS = 1.5
MAX_SHORT_RUN_WORDS = 4

# Combined-score weights. Linguistic and position carry the most weight
# since they're the most reliable signals for a run this short; acoustic
# embedding is supporting evidence only (see module docstring) — too
# little audio for it to be trustworthy alone.
WEIGHT_LINGUISTIC = 0.35
WEIGHT_POSITION = 0.30
WEIGHT_PAUSE = 0.15
WEIGHT_ACOUSTIC = 0.20
# A run must clear this combined score to be reassigned — deliberately
# high so this only fires on strong, multi-signal agreement, never one
# weak signal alone.
REASSIGN_THRESHOLD = 0.65
# A pause at least this long before the run is treated as maximally
# supportive of a genuine turn change.
FULL_PAUSE_SECONDS = 1.0


def _linguistic_score(text: str) -> float:
    """Asks the already-loaded LLM whether `text` reads as a brief
    acknowledgment/backchannel response, as opposed to a substantive
    continuation of the same thought. A real language-model judgment,
    not a hardcoded word list — generalizes across languages and
    phrasing a fixed English vocabulary never would. Only called for
    candidates that already passed the structural short/sandwiched
    checks, so this fires rarely (at most a handful of times per call),
    keeping the per-job cost bounded despite the LLM round-trip."""
    normalized = text.strip()
    if not normalized:
        return 0.0
    try:
        llm = get_llm()
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A speech-to-text transcript produced this short utterance, "
                        f'spoken on its own: "{normalized}"\n\n'
                        "Does this read as a brief acknowledgment or backchannel "
                        "response (an agreement, confirmation, or filler reply), as "
                        "opposed to a substantive statement continuing a thought? "
                        'Respond with only "yes" or "no".'
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=5,
        )
        answer = response["choices"][0]["message"]["content"].strip().lower()
        return 1.0 if answer.startswith("yes") else 0.0
    except Exception:
        logger.exception("linguistic short-run classification failed; treating as neutral")
        return 0.0


def _pause_score(gap_seconds: float) -> float:
    return max(0.0, min(1.0, gap_seconds / FULL_PAUSE_SECONDS))


def _acoustic_score(
    embedding_model,
    audio: np.ndarray,
    sample_rate: int,
    run_start: float,
    run_end: float,
    current_ref: tuple[float, float] | None,
    other_ref: tuple[float, float] | None,
) -> float:
    """(similarity to the OTHER speaker) - (similarity to the CURRENT
    speaker), rescaled around 0.5 = no preference either way. Neutral
    (0.5) whenever a reliable comparison isn't possible — this signal
    should never manufacture confidence that isn't there."""
    if embedding_model is None or current_ref is None or other_ref is None:
        return 0.5
    run_embedding = embed_span(embedding_model, audio, sample_rate, run_start, run_end)
    current_embedding = embed_span(embedding_model, audio, sample_rate, *current_ref)
    other_embedding = embed_span(embedding_model, audio, sample_rate, *other_ref)
    if run_embedding is None or current_embedding is None or other_embedding is None:
        return 0.5
    similarity_to_current = float(np.dot(run_embedding, current_embedding))
    similarity_to_other = float(np.dot(run_embedding, other_embedding))
    return max(0.0, min(1.0, 0.5 + (similarity_to_other - similarity_to_current)))


def _find_nearest_run_with_speaker(
    runs: list[tuple[int, int, str]], from_index: int, speaker: str
) -> tuple[int, int, str] | None:
    """Searches outward from from_index (excluding it) for the nearest run
    matching `speaker` — used to find a same-speaker reference span for
    acoustic comparison when the immediately adjacent runs are (by
    construction, for a sandwiched candidate) the OTHER speaker."""
    for offset in range(1, len(runs)):
        for candidate in (from_index - offset, from_index + offset):
            if 0 <= candidate < len(runs) and runs[candidate][2] == speaker:
                return runs[candidate]
    return None


def _combined_score(
    text: str,
    position: float,
    gap_before: float,
    embedding_model,
    audio: np.ndarray | None,
    sample_rate: int | None,
    can_embed: bool,
    span_start: float,
    span_end: float,
    current_ref: tuple[float, float] | None,
    other_ref: tuple[float, float] | None,
) -> float:
    linguistic = _linguistic_score(text)
    pause = _pause_score(gap_before)
    acoustic = 0.5
    if can_embed:
        acoustic = _acoustic_score(embedding_model, audio, sample_rate, span_start, span_end, current_ref, other_ref)
    return (
        WEIGHT_LINGUISTIC * linguistic
        + WEIGHT_POSITION * position
        + WEIGHT_PAUSE * pause
        + WEIGHT_ACOUSTIC * acoustic
    )


def resolve_short_runs(
    words: list[Word],
    word_speakers: list[str],
    embedding_model,
    audio: np.ndarray | None,
    sample_rate: int | None,
) -> list[str]:
    """Returns a new word_speakers list with high-combined-score short
    spans reassigned to the surrounding speaker. Does not mutate the
    input list. Checks two distinct patterns:

    1. A whole run that's ALREADY short, sandwiched between two runs of
       one other speaker (e.g. a standalone "Just check that." between
       two turns of the other party).
    2. The TRAILING sub-span of a longer run, immediately followed by a
       different speaker's run — catches a short confirming phrase
       glued onto the END of an otherwise-correct longer turn (the
       "...Perfect." case: "Perfect." is not its own run at all, it's
       the tail of a much longer run, so pattern 1 alone never
       considers it). Only ever trims off a SUFFIX, never the whole
       run — a run that's already short enough is pattern 1's job.
    3. The LEADING sub-span of a longer run, immediately preceded by a
       different speaker's run — the mirror of pattern 2. Verified on
       real audio: after long_turn_refiner.py finds a genuine internal
       boundary and reclusters each side against speaker centroids, a
       very short leading piece (too little audio for the acoustic
       centroid alone to classify confidently, see DECISIONS.md #20)
       can still land on the wrong side and get collapsed into the
       following, larger, wrong-speaker segment — at that point it's
       the HEAD of a long run, not its own run, so neither pattern 1
       nor pattern 2 ever considers it. Only ever trims off a PREFIX."""
    if not words:
        return list(word_speakers)

    word_speakers = list(word_speakers)
    n = len(words)

    runs: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        j = i
        while j < n and word_speakers[j] == word_speakers[i]:
            j += 1
        runs.append((i, j, word_speakers[i]))
        i = j

    can_embed = embedding_model is not None and audio is not None and sample_rate is not None

    # Pattern 1: whole short run sandwiched between one other speaker.
    for idx in range(1, len(runs) - 1):
        start_i, end_i, speaker = runs[idx]
        prev_run = runs[idx - 1]
        next_run = runs[idx + 1]

        other_speaker = prev_run[2]
        if other_speaker != next_run[2] or other_speaker == speaker or other_speaker == "unknown":
            continue

        run_start = words[start_i]["start"]
        run_end = words[end_i - 1]["end"]
        if run_end - run_start > MAX_SHORT_RUN_SECONDS or end_i - start_i > MAX_SHORT_RUN_WORDS:
            continue

        run_text = "".join(words[k]["word"] for k in range(start_i, end_i))
        gap_before = run_start - words[start_i - 1]["end"] if start_i > 0 else 0.0

        current_ref = None
        current_run = _find_nearest_run_with_speaker(runs, idx, speaker)
        if current_run:
            current_ref = (words[current_run[0]]["start"], words[current_run[1] - 1]["end"])
        other_ref = (words[prev_run[0]]["start"], words[prev_run[1] - 1]["end"])

        combined = _combined_score(
            run_text, 1.0, gap_before, embedding_model, audio, sample_rate, can_embed,
            run_start, run_end, current_ref, other_ref,
        )
        if combined >= REASSIGN_THRESHOLD:
            for k in range(start_i, end_i):
                word_speakers[k] = other_speaker

    # Pattern 2: trailing sub-span of a longer run, followed by a
    # different speaker. Evaluated against the ORIGINAL runs (computed
    # once above), same as pattern 1 — a run reassigned by pattern 1
    # already became fully short, so this naturally doesn't re-trim it
    # (max_k requires at least 1 word left behind in the run).
    for idx in range(len(runs) - 1):
        start_i, end_i, speaker = runs[idx]
        next_run = runs[idx + 1]
        other_speaker = next_run[2]
        run_len = end_i - start_i
        if other_speaker == speaker or other_speaker == "unknown" or run_len < 2:
            continue

        # Grow the trailing candidate word-by-word while it still fits
        # the short-span limits, keeping the largest valid span.
        max_k = min(MAX_SHORT_RUN_WORDS, run_len - 1)
        candidate_start = None
        for k in range(1, max_k + 1):
            trial_start = end_i - k
            if words[end_i - 1]["end"] - words[trial_start]["start"] > MAX_SHORT_RUN_SECONDS:
                break
            candidate_start = trial_start
        if candidate_start is None:
            continue

        span_start = words[candidate_start]["start"]
        span_end = words[end_i - 1]["end"]
        span_text = "".join(words[k]["word"] for k in range(candidate_start, end_i))
        # Pause WITHIN the run, right before the candidate suffix starts
        # — a real pause there supports treating it as a separate
        # utterance from the rest of the (correctly-assigned) run.
        gap_before = span_start - words[candidate_start - 1]["end"] if candidate_start > 0 else 0.0

        current_ref = (words[start_i]["start"], words[candidate_start - 1]["end"]) if candidate_start > start_i else None
        other_ref = (words[next_run[0]]["start"], words[next_run[1] - 1]["end"])

        combined = _combined_score(
            span_text, 1.0, gap_before, embedding_model, audio, sample_rate, can_embed,
            span_start, span_end, current_ref, other_ref,
        )
        if combined >= REASSIGN_THRESHOLD:
            for k in range(candidate_start, end_i):
                word_speakers[k] = other_speaker

    # Pattern 3: leading sub-span of a longer run, preceded by a
    # different speaker — mirror of pattern 2. Also evaluated against
    # the ORIGINAL runs, so a run pattern 1 already reassigned in full
    # doesn't get re-trimmed here.
    for idx in range(1, len(runs)):
        start_i, end_i, speaker = runs[idx]
        prev_run = runs[idx - 1]
        other_speaker = prev_run[2]
        run_len = end_i - start_i
        if other_speaker == speaker or other_speaker == "unknown" or run_len < 2:
            continue

        # Grow the leading candidate word-by-word while it still fits
        # the short-span limits, keeping the largest valid span.
        max_k = min(MAX_SHORT_RUN_WORDS, run_len - 1)
        candidate_end = None
        for k in range(1, max_k + 1):
            trial_end = start_i + k
            if words[trial_end - 1]["end"] - words[start_i]["start"] > MAX_SHORT_RUN_SECONDS:
                break
            candidate_end = trial_end
        if candidate_end is None:
            continue

        span_start = words[start_i]["start"]
        span_end = words[candidate_end - 1]["end"]
        span_text = "".join(words[k]["word"] for k in range(start_i, candidate_end))
        gap_before = span_start - words[start_i - 1]["end"] if start_i > 0 else 0.0

        current_ref = (words[candidate_end]["start"], words[end_i - 1]["end"]) if candidate_end < end_i else None
        other_ref = (words[prev_run[0]]["start"], words[prev_run[1] - 1]["end"])

        combined = _combined_score(
            span_text, 1.0, gap_before, embedding_model, audio, sample_rate, can_embed,
            span_start, span_end, current_ref, other_ref,
        )
        if combined >= REASSIGN_THRESHOLD:
            for k in range(start_i, candidate_end):
                word_speakers[k] = other_speaker

    return word_speakers

"""``code_opt.reward.code_extraction.extract_code``.

This is the first thing the reward does to every rollout, so it decides what gets
compiled, gated and timed. A change here silently redefines the reward, which is why
the behaviour is pinned in detail -- including the edge cases where the current rules
produce something surprising.

Pure string handling: no compiler, no gem5, no model.
"""
from __future__ import annotations

import pytest

from code_opt.reward.code_extraction import ANSWER_ANCHOR, extract_code


HELLO = 'int main(){ printf("hi"); return 0; }'


# ---------------------------------------------------------------------------
# The normal shape
# ---------------------------------------------------------------------------

def test_fenced_block_after_the_anchor():
    text = f"{ANSWER_ANCHOR}\n```cpp\n{HELLO}\n```"
    assert extract_code(text) == HELLO


def test_prose_around_the_block_is_dropped():
    """Models narrate. The fence, not the surrounding text, delimits the answer."""
    text = (f"Sure, here is a faster version.\n{ANSWER_ANCHOR}\n"
            f"```cpp\n{HELLO}\n```\nThis avoids the extra allocation.")
    assert extract_code(text) == HELLO


@pytest.mark.parametrize("info_string", ["", "cpp", "c++", "C++", "  cpp"])
def test_the_fence_info_string_is_not_part_of_the_code(info_string):
    """The extractor slices between the fence LINES, so whatever language tag the
    model wrote never reaches the compiler."""
    text = f"{ANSWER_ANCHOR}\n```{info_string}\n{HELLO}\n```"
    assert extract_code(text) == HELLO


# ---------------------------------------------------------------------------
# Multiple fences
# ---------------------------------------------------------------------------

def test_last_complete_fence_wins():
    """Models routinely emit the original, an intermediate attempt and then the
    answer. Grading anything but the last complete block would score a draft."""
    text = (f"{ANSWER_ANCHOR}\nFirst attempt:\n```cpp\nint first(){{}}\n```\n"
            f"Actually, better:\n```cpp\nint second(){{}}\n```")
    assert extract_code(text) == "int second(){}"


def test_three_complete_fences_still_take_the_last():
    text = (f"{ANSWER_ANCHOR}\n```cpp\nA\n```\n```cpp\nB\n```\n```cpp\nC\n```")
    assert extract_code(text) == "C"


def test_a_trailing_unterminated_fence_empties_the_result():
    """Recorded because it is surprising and it is what the code does: the rule is
    'the span between the last TWO fence lines', not 'the last balanced pair'. A
    completion truncated mid-way through re-opening a block therefore extracts the
    empty string -- which the reward turns into a score-0 'no C++ code' rollout
    rather than grading the earlier, complete block."""
    text = f"{ANSWER_ANCHOR}\n```cpp\nA\n```\n```cpp\nB_truncated"
    assert extract_code(text) == ""


def test_a_single_unterminated_fence_keeps_its_marker():
    """Same rule seen from the other side: with only ONE fence line there is no
    pair to slice between, so the post-anchor body is returned verbatim -- backtick
    line included. It will not compile, so such a rollout scores 0."""
    text = f"{ANSWER_ANCHOR}\n```cpp\n{HELLO}"
    assert extract_code(text) == f"```cpp\n{HELLO}"


# ---------------------------------------------------------------------------
# Special tokens and reasoning traces
# ---------------------------------------------------------------------------

def test_chat_special_tokens_are_stripped():
    """Present whenever the sampler decodes with skip_special_tokens=False; they
    would otherwise end up inside the compiled source."""
    text = (f"<|im_start|>assistant\n{ANSWER_ANCHOR}\n"
            f"```cpp\n{HELLO}\n```<|im_end|>")
    assert extract_code(text) == HELLO


@pytest.mark.parametrize("token", ["<|endoftext|>", "<|im_end|>", "<|tool_call|>"])
def test_any_pipe_delimited_special_token_is_stripped(token):
    text = f"{token}{ANSWER_ANCHOR}\n```cpp\n{HELLO}\n```{token}"
    assert extract_code(text) == HELLO


def test_think_prefix_is_dropped_including_any_code_inside_it():
    """Reasoning models draft code inside <think>. That draft is not the answer,
    and a fence inside the trace must not be able to win the 'last fence' rule."""
    text = ("<think>\nLet me try ```cpp\nint draft(){}\n``` first.\n</think>\n"
            f"{ANSWER_ANCHOR}\n```cpp\n{HELLO}\n```")
    assert extract_code(text) == HELLO


def test_think_with_no_closing_tag_is_left_alone():
    """Only a CLOSED </think> triggers the split, so a truncated trace does not
    silently discard the whole completion by accident."""
    text = f"<think>\nreasoning\n{ANSWER_ANCHOR}\n```cpp\n{HELLO}\n```"
    assert extract_code(text) == HELLO


# ---------------------------------------------------------------------------
# The anchor -- the case that matters most
# ---------------------------------------------------------------------------

def test_a_preamble_echo_of_the_slow_program_is_not_graded():
    """The case the anchor exists for. A model that restates the SLOW source before
    answering must not have that echo compiled and timed: it passes correctness and
    scores a speedup of ~1.0, which looks like a real (if useless) reward and would
    quietly cap the whole run."""
    text = ("Here is the program you gave me:\n"
            "```cpp\nint slow(){ /* the original */ }\n```\n"
            f"{ANSWER_ANCHOR}\n```cpp\nint fast(){{}}\n```")
    assert extract_code(text) == "int fast(){}"


def test_everything_before_the_anchor_is_discarded_even_without_a_fence():
    text = f"blah blah\n{ANSWER_ANCHOR}\n{HELLO}"
    assert extract_code(text) == HELLO


def test_the_first_anchor_wins_when_the_model_repeats_it():
    """split(anchor, 1) keeps everything after the FIRST occurrence, so a repeated
    anchor still leaves the final fenced block as the last complete one."""
    text = (f"{ANSWER_ANCHOR}\nlet me redo that\n{ANSWER_ANCHOR}\n"
            f"```cpp\n{HELLO}\n```")
    assert extract_code(text) == HELLO


def test_no_fence_at_all_returns_the_whole_post_anchor_body():
    """The anchor-prefilled prompt makes bare code the expected shape, not an
    error -- so an unfenced answer is graded, not thrown away."""
    body = "#include <cstdio>\nint main(){ return 0; }"
    assert extract_code(f"{ANSWER_ANCHOR}\n{body}") == body


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n", "<|im_start|>", "</think>"])
def test_empty_or_token_only_input_returns_the_empty_string(text):
    """Falsy, which is what the reward's ``if not code or not code.strip()`` guard
    turns into a phase='extract' zero rather than a compiler invocation."""
    assert extract_code(text) == ""


def test_garbage_prose_is_returned_verbatim_not_rejected():
    """Actual behaviour, recorded rather than assumed: with no anchor and no fence
    the extractor is not a validator -- it hands the prose downstream, where the
    compile step fails and the rollout scores 0. No exception, no crash."""
    text = "I am sorry, but I cannot optimize this program."
    assert extract_code(text) == text


def test_extract_code_always_returns_a_string():
    """The reward calls ``.strip()`` on the result unconditionally, so None would
    be an AttributeError inside the grader rather than a zero-scored rollout."""
    for text in ["", "x", f"{ANSWER_ANCHOR}", "```", "```\n```"]:
        assert isinstance(extract_code(text), str)

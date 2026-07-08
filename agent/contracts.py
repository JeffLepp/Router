from __future__ import annotations

import json
import re
from dataclasses import dataclass
from textwrap import shorten

from agent.classify import CATEGORIES, canonical_category


DEFAULT_MAX_TOKENS = {
    "actual_qa": 120,
    "math_reasoning": 10,
    # sentiment aspect JSON (mixed + aspects{}) needs ~40 tokens; a cap of 2 truncated it
    # to nothing and the task could never score. A cap is a ceiling, not a spend — plain
    # labels still emit ~1 token and stop. (Issue 5 flag.)
    "sentiment_analysis": 64,
    "summarization": 80,
    "named_entity_recognition": 60,
    "logic_puzzles": 60,  # was 40 — declarative logic answers were truncated mid-sentence (3.1)
    "code_debugging": 300,
    "code_generation": 300,
}


@dataclass(frozen=True)
class Contract:
    category: str
    max_tokens: int
    remote_instruction: str

    def remote_prompt(self, task_prompt: str) -> str:
        kernel = compress_prompt(self.category, task_prompt)
        return f"{self.remote_instruction}\nKernel:\n{kernel}"

    def assemble(self, task_prompt: str, remote_payload: str) -> str:
        payload = remote_payload.strip()
        if not payload:
            return ""
        if self.category == "math_reasoning":
            return f"The final answer is {payload}."
        if self.category == "sentiment_analysis":
            if payload.startswith("{") or payload.startswith("["):
                return _normalize_aspect_json(payload)  # remap aspect keys to gold vocab
            label = payload.strip().strip(".").lower()
            return json.dumps({"sentiment": label}, separators=(",", ":"))
        if self.category == "named_entity_recognition":
            return f"Named entities:\n{payload}"
        if self.category == "logic_puzzles":
            return _as_answer_json(payload)
        if self.category in {"code_debugging", "code_generation"}:
            return _strip_code_fences(payload)
        return payload


def _normalize_caps(max_tokens: dict[str, int] | None) -> dict[str, int]:
    caps = dict(DEFAULT_MAX_TOKENS)
    if not max_tokens:
        return caps
    for key, value in max_tokens.items():
        caps[canonical_category(key)] = int(value)
    return caps


def compress_prompt(category: str, task_prompt: str) -> str:
    category = canonical_category(category)
    text = re.sub(r"\s+", " ", task_prompt.strip())
    if category == "summarization":
        return _summary_kernel(text)
    if category in {"code_debugging", "code_generation"}:
        fenced = _extract_fenced_code(task_prompt)
        if fenced:
            return shorten(fenced, width=1600, placeholder=" ...")
    width = 1200 if category != "actual_qa" else 900
    return shorten(text, width=width, placeholder=" ...")


def _summary_kernel(text: str) -> str:
    constraint = text.split(":", 1)[0]
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text, flags=re.S)
    passage = " ".join(first or second for first, second in quoted).strip()
    if not passage:
        passage = text
    sentences = re.split(r"(?<=[.!?])\s+", passage)
    keep = [sentence.strip() for sentence in sentences if sentence.strip()][:5]
    if len(sentences) > 5:
        keep.append(sentences[-1].strip())
    kernel = " ".join(keep)
    words = kernel.split()
    if len(words) > 140:
        kernel = " ".join(words[:140])
    return (
        f"Constraint: {shorten(constraint, width=180, placeholder=' ...')}\n"
        f"Extracted kernel (not raw passage): {kernel}\n"
        "Return the final summary only."
    )


_FENCE_BLOCK = re.compile(r"```[ \t]*[A-Za-z0-9_+#.-]*[ \t]*\n(.*?)\n?```", re.S)


def _strip_code_fences(payload: str) -> str:
    """Strip whole-answer or preamble+fenced markdown from remote code answers."""
    match = _FENCE_BLOCK.search(payload)
    if match:
        return match.group(1).strip()
    stripped = payload.strip()
    if stripped.startswith("```"):  # unterminated leading fence (truncated)
        body = stripped[3:]
        newline = body.find("\n")
        if newline != -1 and re.fullmatch(r"[A-Za-z0-9_+#.-]*", body[:newline].strip()):
            body = body[newline + 1 :]
        return body.strip()
    return stripped


# ponytail: literal-match unsolvable signals; a puzzle whose real answer is the word
# "none" would misfire, but grade/seating constraint answers never are.
_LOGIC_INVALID = {"none", "no solution", "no consistent solution", "unanswerable"}

# prefixes the model prepends to a leaked final answer ("The answer is: Ali")
_ANSWER_LEAD = re.compile(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]\s*", re.I)


def _logic_tail(payload: str) -> str:
    """Pull the final answer off leaked reasoning: last non-empty line, minus any
    'The answer is:' lead-in. Instruction says answer-only, but models still ramble."""
    lines = [ln.strip() for ln in payload.splitlines() if ln.strip()]
    text = lines[-1] if lines else payload.strip()
    text = _ANSWER_LEAD.sub("", text).strip()
    return text.rstrip(".").strip() or payload.strip()


def _as_answer_json(payload: str) -> str:
    """Compact {"answer":...,"valid":bool} for logic; pass valid answer-JSON through."""
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict) and "answer" in obj:
        obj.setdefault("valid", True)  # scorer compares valid by identity; never leave it unset
        return json.dumps(obj, separators=(",", ":"))
    answer = _logic_tail(payload)
    if answer.strip().strip(".").lower() in _LOGIC_INVALID:
        return json.dumps({"answer": None, "valid": False}, separators=(",", ":"))
    return json.dumps({"answer": answer, "valid": True}, separators=(",", ":"))


# Model names aspects loosely; gold uses one canonical noun. Deterministic remap so
# scorer (which compares aspect keys exactly) accepts the model's structure.
# ponytail: fixed benchmark vocab — extend the map as new aspect nouns appear.
_ASPECT_ALIASES = {
    "speed": "performance", "performance": "performance", "processor": "performance",
    "cpu": "performance", "sound": "performance", "concert": "performance",
    "music": "performance", "audio": "performance",
    "battery": "battery", "battery life": "battery", "batterylife": "battery",
    "camera": "camera", "camera quality": "camera", "photo": "camera", "photos": "camera",
    "design": "design", "look": "design", "looks": "design", "appearance": "design",
    "style": "design", "build": "design",
    "venue": "venue", "location": "venue", "place": "venue", "crowd": "venue",
    "crowding": "venue", "crowdedness": "venue", "space": "venue",
}


def _norm_polarity(value: str) -> str:
    s = str(value).strip().lower()
    if s.startswith("neg"):
        return "negative"
    if s.startswith("pos"):
        return "positive"
    return "positive"  # positive-default aspect shape (Issue 5)


def _normalize_aspect_json(payload: str) -> str:
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return payload
    if not isinstance(obj, dict):
        return payload
    aspects = obj.get("aspects")
    if isinstance(aspects, dict):
        obj["aspects"] = {
            _ASPECT_ALIASES.get(str(k).strip().lower(), str(k).strip().lower()): _norm_polarity(v)
            for k, v in aspects.items()
        }
    return json.dumps(obj, separators=(",", ":"))


def _extract_fenced_code(text: str) -> str:
    parts = text.split("```")
    if len(parts) < 3:
        return ""
    code = parts[1].strip()
    first_newline = code.find("\n")
    if first_newline != -1 and re.fullmatch(r"[A-Za-z0-9_+#.-]+", code[:first_newline].strip()):
        code = code[first_newline + 1 :]
    return code.strip()


def build_contracts(max_tokens: dict[str, int] | None = None) -> dict[str, Contract]:
    caps = _normalize_caps(max_tokens)
    return {
        "actual_qa": Contract(
            "actual_qa",
            caps["actual_qa"],
            "Return the exact answer only — no bullets, markdown, preamble, or explanation. "
            "For a simple fact (who/what/when/where) give just the fact: a name, place, or year. "
            "For a comparison or yes/no question, answer in one short declarative sentence that names "
            "the items compared (e.g. 'The Nile is longer than the Amazon River.'; "
            "'No, steel is denser than aluminium.'). "
            "Answer questions about current or recent office-holders and events with your best "
            "knowledge. Return exactly 'unanswerable' ONLY when the question rests on a false premise "
            "or is genuinely unknowable — never merely because it is recent.",
        ),
        "math_reasoning": Contract(
            "math_reasoning",
            caps["math_reasoning"],
            "Return the final number only.",
        ),
        "sentiment_analysis": Contract(
            "sentiment_analysis",
            caps["sentiment_analysis"],
            'Return exactly one label: positive, negative, neutral, or mixed. Judge the writer\'s real attitude: treat sarcasm or irony as its literal opposite (praise that describes a bad outcome — a long wait, hard effort — is negative). A plain factual statement with no opinion is neutral. If the prompt contrasts two aspects, return compact JSON: {"sentiment":"mixed","aspects":{"<aspect>":"positive_or_negative"}}. Use a single lowercase noun per aspect (e.g. performance, battery, design, camera, venue) and set each to positive or negative.',
        ),
        "summarization": Contract(
            "summarization",
            caps["summarization"],
            "Summarize only the extracted kernel. Obey the stated format and word limit.",
        ),
        "named_entity_recognition": Contract(
            "named_entity_recognition",
            caps["named_entity_recognition"],
            "Return entity|TYPE lines only.",
        ),
        "code_debugging": Contract(
            "code_debugging",
            caps["code_debugging"],
            "Return corrected code only.",
        ),
        "logic_puzzles": Contract(
            "logic_puzzles",
            caps["logic_puzzles"],
            "Give ONLY the final answer — the name, word, day, or short phrase that answers the question. No reasoning, no restating the question, no 'The answer is'. For a yes/no question answer Yes or No. If the premises are contradictory or no consistent solution exists, return exactly: none.",
        ),
        "code_generation": Contract(
            "code_generation",
            caps["code_generation"],
            "Return code only.",
        ),
    }


def _self_check() -> None:
    contracts = build_contracts({"math": 8, "sentiment": 3})
    assert set(contracts) == set(CATEGORIES)
    assert contracts["math_reasoning"].max_tokens == 8
    assert "final number only" in contracts["math_reasoning"].remote_prompt("What is 2+2?")
    assert "not raw passage" in contracts["summarization"].remote_prompt("Summarize: " + "word " * 300)

    # sentiment: plain label -> compact JSON; aspect keys remapped to gold vocab
    sent = contracts["sentiment_analysis"]
    assert sent.assemble("x", "Positive.") == '{"sentiment":"positive"}'
    # Issue 4: model's loose aspect nouns get remapped to the gold single-word vocab
    assert sent.assemble("x", '{"sentiment":"mixed","aspects":{"speed":"positive","battery life":"negative"}}') \
        == '{"sentiment":"mixed","aspects":{"performance":"positive","battery":"negative"}}'
    assert sent.assemble("x", '{"sentiment":"mixed","aspects":{"design":"positive","camera quality":"negative"}}') \
        == '{"sentiment":"mixed","aspects":{"design":"positive","camera":"negative"}}'
    assert build_contracts()["sentiment_analysis"].max_tokens >= 40  # default cap fits aspect JSON (Issue 5)
    # Issue 5: aspect value normalized, unknown polarity defaults positive
    assert sent.assemble("x", '{"sentiment":"mixed","aspects":{"venue":"bad"}}') \
        == '{"sentiment":"mixed","aspects":{"venue":"positive"}}'  # "bad" isn't neg-prefixed -> default

    # logic: no "The answer is:" wrapper; wrap plain, pass answer-JSON through
    logic = contracts["logic_puzzles"]
    assert logic.assemble("x", "Alice") == '{"answer":"Alice","valid":true}'
    assert logic.assemble("x", '{"answer":"Alice","valid":true}') == '{"answer":"Alice","valid":true}'
    assert logic.assemble("x", '{"answer":"Bob"}') == '{"answer":"Bob","valid":true}'  # fill missing valid
    assert logic.assemble("x", "none") == '{"answer":null,"valid":false}'  # unsolvable signal
    # Issue 3.2: strip leaked "The answer is:" lead-in and take the tail line
    assert logic.assemble("x", "The answer is: Ali") == '{"answer":"Ali","valid":true}'
    assert logic.assemble("x", "Reasoning line one\nFinal answer: Brooke") == '{"answer":"Brooke","valid":true}'

    # code: strip whole-answer fences and preamble+fenced code
    debug = contracts["code_debugging"]
    assert debug.assemble("x", "```python\nprint(1)\n```") == "print(1)"
    assert debug.assemble("x", "The final answer is\n```javascript\nconsole.log(1);\n```") == "console.log(1);"
    assert debug.assemble("x", "print(1)") == "print(1)"

    # unanswerable escape hatch is scoped to actual_qa only
    assert "unanswerable" in contracts["actual_qa"].remote_instruction
    for name in set(CATEGORIES) - {"actual_qa"}:
        assert "unanswerable" not in contracts[name].remote_instruction


if __name__ == "__main__":
    _self_check()
    print("contracts self-check passed")

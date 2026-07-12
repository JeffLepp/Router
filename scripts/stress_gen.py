"""Generate eval/devset/stress.json: adversarial stress set for the zero-token surfaces.

Two kinds of entries:
- positive: paraphrase/parameter sweeps of shapes the gate templates claim to cover.
  Gold answers computed from true semantics; a solver may answer (must match) or defer.
- trap: near-miss prompts whose surface form matches a template but whose true answer
  differs. The gate/prefilter must defer; a confident answer here is a hidden-set miss.

Deterministic (no RNG). Run: python -m scripts.stress_gen
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

OUT = Path("eval/devset/stress.json")

tasks: list[dict] = []


def fmt(v) -> str:
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    f = float(v)
    return str(int(f)) if f == int(f) else str(f)


def add(category: str, prompt: str, gold, kind: str, note: str = "") -> None:
    tasks.append(
        {
            "id": f"s_{category.split('_')[0]}_{len(tasks) + 1:03d}",
            "category": category,
            "difficulty": "hard" if kind == "trap" else "medium",
            "holdout": True,
            "evaluation_method": "exact_match" if gold is not None else "unanswerable",
            "requires_tools": False,
            "recommended_tool": None,
            "prompt": prompt,
            "expected_answer": gold,
            "stress_kind": kind,
            "note": note,
        }
    )


M = "math_reasoning"

# --- math positives: same semantics as the templates, varied surface -------------
for rate, hours, subj, q in [
    (55, 2, "A delivery van drives at 55 km/h for 2 hours.", "How far does it go?"),
    (16, 2.5, "A cyclist pedals at a constant 16 km/h for 2.5 hours.", "How many kilometers does she cover?"),
    (480, 3, "A plane flies at 480 km/h for 3 hours.", "How far does it travel?"),
]:
    add(M, f"{subj} {q}", fmt(rate * hours), "positive")
add(M, "A bus covers 180 km in 3 hours. How many kilometers per hour is it traveling?", "60", "positive")
add(M, "A runner covers 24 km in 2 hours. What is her speed in km/h?", "12", "positive")
for pct, base in [(12, 150), (7.5, 400)]:
    add(M, f"What is {pct}% of {base}?", fmt(pct * base / 100), "positive")
add(M, "What is 17 x 6?", "102", "positive")
add(M, "What is 144 / 12?", "12", "positive")
add(M, "Calculate 15 + 27 * 2.", "69", "positive")
add(M, "What is 3 raised to the fifth power?", "243", "positive")
add(M, "What is 12 squared?", "144", "positive")
add(M, "What is the cube of 7?", "343", "positive")
add(M, "Take the number 9, triple it, then add 4. What number results?", "31", "positive")
add(M, "Start with 40, halve it, then subtract 6. What is the result?", "14", "positive")
add(M, "What is the average of 12, 18, and 24?", "18", "positive")
add(M, "What is the average of 5, 10, 15, and 20?", "12.5", "positive")
add(M, "Ravi had 60 marbles, gave 24 to a friend, and later bought 9 more. How many marbles does he have now?", "45", "positive")
add(M, "Ella bought 4 notebooks at $3 each and 2 pens at $1.50 each. How much did she spend in total?", "15", "positive")
add(M, "A $250 desk has a 30% discount. What is the sale price?", "175", "positive")
add(M, "A kettle costs $63 after a 10% discount. What was the original price?", "70", "positive")
add(M, "Convert 3.2 kilometers to meters.", "3200", "positive")
add(M, "Convert 450 centimeters to meters.", "4.5", "positive")
add(M, "Find all real solutions x to the quadratic equation x^2 - 7x + 12 = 0.", "[3, 4]", "positive")
add(M, "A jar contains 4 red and 6 blue marbles. If you draw 2 marbles without replacement, what is the probability that both are blue?", "1/3", "positive")
add(M, "If a rectangle has a length of 12 cm and a width of 7 cm, what is its area?", "84", "positive")
add(M, "A patio measures 9 meters by 4 meters. How many square meters does it cover?", "36", "positive")
add(M, "What is 45 divided by zero?", None, "positive")
add(M, "A triangle has side lengths 1 cm, 2 cm, and 5 cm. What is its area?", None, "positive")

# --- math traps: template fires, true answer differs ------------------------------
add(M, "A car travels at 60 km/h for 2 hours. How many meters does it cover?",
    "120000", "trap", "rate*time answers 120 (km) but the question asks meters")
add(M, "A car travels at 60 km/h for 2 hours to reach the coast. What was its average speed in km/h?",
    "60", "trap", "rate*time answers distance 120; question asks the speed")
add(M, "Jane bought 3 books at $12 each and paid with a $50 bill. How much change did she receive in total?",
    "14", "trap", "item_total answers 36, ignoring the change question")
add(M, "Mark bought 4 mugs at $9 each and used a $10 gift card. What total did he pay?",
    "26", "trap", "item_total answers 36, ignoring the gift card")
add(M, "A rectangle has a length of 8 cm and a width of 5 cm. A second rectangle covers double that area. What is the area of the second rectangle?",
    "80", "trap", "rectangle_area answers 40 for the wrong rectangle")
add(M, "Nina had 45 stickers, gave 17 to her brother, and later found 8 more in a drawer. How many stickers did her brother receive?",
    "17", "trap", "had_gave_found answers 36; question asks about the brother")
add(M, "A $100 jacket has a 20% discount, and 5% sales tax is added to the sale price. What is the final cost?",
    "84", "trap", "discount_price answers 80, ignoring the tax")
add(M, "Tom bought a bike for $80 and sold it for $100. What was his profit as a percentage of the price he paid?",
    "25", "trap", "profit_loss answers 20 (absolute), question asks percent")
add(M, "Two coats together cost $300 after a 25% discount was applied to each. What was the original price of one coat?",
    "200", "trap", "reverse_percent answers 400 for the pair, not one coat")
add(M, "What is 15% of 200, plus 30?",
    "60", "trap", "percent_of answers 30, dropping the trailing addition")
add(M, "What is the square of 9, minus 1?",
    "80", "trap", "square_of answers 81, dropping the trailing subtraction")
add(M, "Take the number 12, double it, then subtract five. What number results?",
    "19", "trap", "op_chain misses the word-number step and answers 24")
add(M, "What is the average of 4, 8, and 9 after removing the largest number?",
    "6", "trap", "average answers 7 over all three numbers")
add(M, "What is 10 / 0.5?",
    "20", "trap", "division-by-zero regex misfires on /0.5 and answers null")
add(M, "Find all real solutions x to the quadratic equation 2x^2 + 5x + 6 = 0.",
    None, "trap", "quadratic template ignores the leading 2 and answers [-3, -2]")
add(M, "A jar holds 3 red and 5 blue marbles. You draw 2 marbles without replacement. What is the probability that both are red or both are blue?",
    "13/28", "trap", "probability answers only the both-red branch 3/28")
add(M, "A train covers 240 km in 4 hours. What is its speed in meters per second?",
    "16.67", "trap", "speed template answers 60 km/h; question asks m/s")
add(M, "A hiker walks at 4 km/h for 2 hours, then rests for 1 hour before continuing. How far has the hiker walked?",
    "8", "trap", "extra-number guard must hold: rest hour is not a leg")
add(M, "A car travels at 60 km/h for 2 hours and then at 40 km/h for 1 hour. How many kilometers does it travel in total?",
    "160", "trap", "multi-leg trip; single-leg template must defer")
add(M, "A rectangle has a length of 8 cm and a width of 5 cm. What is its perimeter?",
    "26", "trap", "perimeter question near the area template")
add(M, "What is 20% off of $150?",
    "120", "trap", "off-of phrasing; percent_of must not misparse")
add(M, "A price of $200 is increased by 10% and then decreased by 10%. What is the final price?",
    "198", "trap", "sequential percent change; no template applies")
add(M, "A car drives 120 km at 60 km/h and returns at 40 km/h. What is its average speed for the whole trip?",
    "48", "trap", "harmonic-mean average speed; must defer")

# --- prefilter traps and controls -------------------------------------------------
add("code_debugging", "Debug why my summary function returns None when the list is empty.",
    "The function likely mutates in place (e.g. list.sort()) or lacks a return for the empty branch.",
    "trap", "summary noun inside a debug ask must not route to summarization")
add("actual_qa", "What is an executive summary?",
    "A short overview of a longer document highlighting its key points.",
    "trap", "summary noun inside a definition question")
add("actual_qa", "What does a Python function return if it has no return statement?",
    "None", "trap", "language+artifact nouns inside a factual question")
add("sentiment_analysis", "The Python workshop was fantastic and the instructor made every function and class feel simple.",
    "positive", "trap", "language+artifact nouns inside a review")
add("sentiment_analysis", "The people at this place were rude, and the following morning the front desk never called us back.",
    "negative", "trap", "entity nouns plus 'following' inside a review")
add("logic_puzzles", "Four people - Ana, Ben, Cara, and Dan - each visit a different city on different dates. Ana and Ben avoided Europe, and Dan visited Rome. Given the following clues, determine who visited Paris.",
    "Cara", "trap", "entity nouns plus 'following' inside a logic puzzle")
add("actual_qa", "Answer yes or no: is the Great Wall of China visible from the Moon with the naked eye?",
    "No", "trap", "'answer yes or no' phrasing on a factual question")
add("code_generation", "Write a Python function that returns a summary dict of word counts for a given string.",
    None, "trap", "'summary dict' artifact inside a code-generation ask")
add("code_generation", "Write a summary function in Python that averages a list of numbers.",
    None, "trap", "artifact literally named 'summary function'")
add("code_generation", "Write a one line Python script that prints the current date and time in ISO format, using only modules from the standard library that ship with CPython 3.12, and include a brief comment naming the module you chose and why it is the right default on Windows terminals.",
    None, "trap", "'one line' shape cue in a long code-generation prompt")
add("logic_puzzles", "Name the people and organizations involved in the following puzzle, then determine who sits where.",
    None, "trap", "extraction verbs plus entity nouns inside a logic puzzle")
add("summarization", "Summarize the following support ticket in one sentence: 'The customer reports that the mobile app closes unexpectedly when uploading photos larger than 10 MB, and support suggested reinstalling without success.'",
    None, "positive", "explicit summarize imperative with incident vocabulary in the body")
add("named_entity_recognition", "Extract all people and organizations mentioned in this sentence: 'Tim Cook met with executives from Nike and Adidas in Portland last Tuesday.'",
    None, "positive")
add("sentiment_analysis", "The pasta arrived lukewarm and the waiter forgot our drinks twice.",
    "negative", "positive")
add("logic_puzzles", "All tulips are flowers and all flowers need water. Does it follow that all tulips need water? Answer yes or no.",
    "Yes", "positive")
add("code_debugging", "This Python function is supposed to return the largest number but it returns the smallest. Fix the bug:\n```python\ndef largest(xs):\n    return min(xs)\n```",
    "return max(xs)", "positive")
add("code_generation", "Implement a JavaScript function that returns the factorial of a non-negative integer n.",
    None, "positive")


def main() -> None:
    OUT.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    kinds = {}
    for t in tasks:
        kinds[t["stress_kind"]] = kinds.get(t["stress_kind"], 0) + 1
    print(f"wrote {OUT} with {len(tasks)} tasks: {kinds}")


if __name__ == "__main__":
    main()

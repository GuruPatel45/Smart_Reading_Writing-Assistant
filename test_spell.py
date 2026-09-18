import sys
sys.path.append('.')
from stage1_spellcheck.spell_checker import SpellChecker

checker = SpellChecker()

tests = [
    "Rahul sent the report to Priya yestarday.",
    "Rahul sent the report to Priya yesterday.",
    "I went to the bank yestarday because I need money.",
    "I recieved a mesage from my professor.",
    "Rahul and Priya completed the project."
]

for t in tests:
    res, diffs = checker.process(t)
    print("IN :", t)
    print("OUT:", res)
    print("DIF:", diffs)
    print("-" * 40)

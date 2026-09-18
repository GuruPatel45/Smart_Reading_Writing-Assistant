from stage1_spellcheck.spell_checker import SpellChecker
checker = SpellChecker()
text = '"I went to the bank yestarday becuse I need money."'
res, diffs = checker.process(text)
print("Corrected:", res)
print("Diffs:", diffs)

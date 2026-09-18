# Building a "Smart Reading & Writing Assistant" (NLP Pipeline)

**Type:** Practical / implementation-based

**Suggested duration:** 3 weeks

**Deliverable:** Working code + 5-minute recorded demo + 1-page README

## 1. The Scenario

Build a lightweight assistant — like a mini Grammarly/Word-processor plugin — that takes a raw paragraph of student-submitted text (emails, assignment answers, forum posts) and processes it end-to-end through the classic NLP pipeline. This is deliberately built as one continuous pipeline so each syllabus topic is a visible, testable stage rather than an isolated exercise.

Test input: provide the tool a folder of 10 short "messy" real-world text samples (typos, ambiguous pronouns, informal grammar) — students may collect these themselves from real forum posts, emails, or chat logs (anonymized).

## 2. Pipeline Stages

### Stage 1 — Spell Checking

**Topic:** Spell Checking

1. Implement a spell checker using **edit distance** (Levenshtein) against a dictionary/word-frequency list — do this from scratch, don't just call a library function, so the underlying algorithm is demonstrated.
2. Add a **context-aware** improvement: for ambiguous corrections (e.g., "there" vs "their"), use surrounding word frequency/n-gram probability to pick the more likely correction rather than just the closest edit distance.
3. Output: corrected text + a diff highlighting exactly which words were changed and why (edit-distance score or n-gram probability).

### Stage 2 — Syntactic Processing

**Topic:** Syntactic Processing

1. Use a parser (spaCy, NLTK, or Stanza) to produce a **dependency parse tree** and a **POS tag sequence** for each sentence.
2. Implement (by hand, not just calling a library) a simple **CYP/CKY or recursive-descent parser** for a small custom grammar (10–15 CFG rules covering simple declarative sentences) to demonstrate you understand parsing mechanics, not just tool usage.
3. Output: a visual parse tree (can use `nltk.tree` rendering or `displacy`) per sentence.

### Stage 3 — Semantic Analysis

**Topic:** Semantic Analysis

1. Extract **named entities** and **semantic roles** (who did what to whom) using spaCy/AllenNLP SRL, or a hand-built rule set over the dependency tree if you want the harder/more instructive route.
2. Build simple **word-sense disambiguation**: for at least 3 ambiguous words in your test set (e.g., "bank," "bat," "light"), use a Lesk-algorithm-style approach (WordNet overlap) to pick the correct sense given context.
3. Output: for each sentence, a structured JSON like `{"agent": "...", "action": "...", "patient": "...", "entities": [...]}`.

### Stage 4 — Discourse & Pragmatic Processing

**Topic:** Discourse and Pragmatic Processing

1. Implement **coreference resolution** across sentences in a paragraph (who does "he"/"it"/"they" refer to?) — use neuralcoref/coreferee/spaCy's coref component, or a simple heuristic resolver (nearest matching antecedent by gender/number agreement) if a pretrained coref model isn't available.
2. Detect **discourse connectives** (e.g., "however," "therefore," "because") and label the relation they signal (contrast, cause, elaboration) using a small hand-built rule list — this demonstrates discourse relation typing at a practical level.
3. Add one **pragmatic inference** rule: e.g., detect indirect requests ("Could you send the file?" → actually means "send me the file," not a yes/no question) using pattern matching over the parsed structure.

### Stage 5 — Full Pipeline Assembly

Wire stages 1→4 into a single function: `process(raw_text) → corrected_text, parse_trees, semantic_frames, discourse_relations, coref_chains`.

Run it on your 10 test samples and produce one summary table showing, per sample: # spelling corrections, # entities found, # coref chains resolved, # discourse relations tagged.

## 3. What to Submit

- Code repo with 5 stage folders + `pipeline.py`
- `README.md`: how to run, plus a max-5-sentence note per stage on what it demonstrates
- 5-minute demo video running the pipeline live on 2–3 of your real-world text samples

## 4. Suggested Tools

Python 3.10+, spaCy or NLTK (POS/dependency parsing, NER), NLTK WordNet (Lesk WSD), coreferee or neuralcoref (coreference), your own from-scratch code for edit-distance spell checking and the CFG parser (these two must not just call a library — that's the point of the exercise).

## 5. Stretch Goal (optional bonus, +10 pts)

Turn Stage 1's spell checker into a tiny autocomplete-as-you-type demo using a simple Streamlit or Flask front end, so the "real world" framing (a writing assistant) is visibly usable, not just a script.

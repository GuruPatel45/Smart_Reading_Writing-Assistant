# Smart Reading & Writing Assistant (NLP Pipeline)

This project implements an end-to-end NLP pipeline acting as a smart writing assistant. It processes raw, "messy" text paragraphs through five distinct stages, showcasing classic natural language processing techniques.

## How to Run

1. **Setup the Virtual Environment and Install Dependencies:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Download NLP Models:**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
   ```

3. **Run the Full Pipeline:**
   ```bash
   python pipeline.py
   ```
   This will process the 10 real-world text samples in the `data/` folder and output a summary table (`summary_table.csv`).

4. **Run the Interactive Autocomplete Demo (Stretch Goal):**
   ```bash
   streamlit run app.py
   ```

## Pipeline Stages

### Stage 1: Spell Checking
Demonstrates a custom spell checker built from scratch using the Levenshtein edit distance algorithm. It evaluates character-level modifications (insertions, deletions, substitutions) to find the closest valid word. Additionally, it implements context-aware correction using n-gram probabilities to select the best candidate when multiple words share the same edit distance. 

### Stage 2: Syntactic Processing
Demonstrates part-of-speech tagging and dependency parsing utilizing the `spaCy` library. It also features a custom, manually-implemented top-down recursive descent parser over a small Context-Free Grammar (CFG). This highlights the fundamental mechanics of parsing beyond simply calling a library function.

### Stage 3: Semantic Analysis
Demonstrates the extraction of semantic roles (agent, action, patient) and named entities using rules over the `spaCy` dependency tree. It also implements the Simplified Lesk algorithm from scratch using WordNet overlap to achieve Word Sense Disambiguation (WSD) for ambiguous target words based on their sentence context.

### Stage 4: Discourse & Pragmatic Processing
Demonstrates coreference resolution using a heuristic resolver that links pronouns to the nearest matching antecedent by gender and plurality. It detects specific discourse connectives (e.g., "however", "therefore") to label rhetorical relations such as contrast, cause, or elaboration. Furthermore, it applies pragmatic inference rules to detect indirect requests.

### Stage 5: Full Pipeline Assembly
Demonstrates the integration of all previous stages into a cohesive, single-pass pipeline function `process(raw_text)`. It processes a batch of 10 real-world, "messy" text samples and generates a structured summary table detailing the spelling corrections, entities found, coreference chains resolved, and discourse relations tagged for each sample.

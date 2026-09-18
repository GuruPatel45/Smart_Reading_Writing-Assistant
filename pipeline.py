import os
import glob
import pandas as pd
from stage1_spellcheck.spell_checker import SpellChecker
from stage2_syntax.syntactic_processor import SyntacticProcessor
from stage3_semantics.semantic_analyzer import SemanticAnalyzer
from stage4_discourse.discourse_processor import DiscourseProcessor

import spacy

class NLPPipeline:
    def __init__(self):
        print("Initializing Pipeline...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        self.spell_checker = SpellChecker()
        self.syntactic_processor = SyntacticProcessor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.discourse_processor = DiscourseProcessor()
        print("Pipeline initialized.")

    def process(self, raw_text):
        """
        Processes text end-to-end through the NLP pipeline.
        """
        # Stage 1
        corrected_text, spell_diff = self.spell_checker.process(raw_text)
        
        # Centralized Parsing & Linguistic Normalization
        # Clean whitespace to prevent SPACE tokens in tree
        clean_text = " ".join(corrected_text.split())
        doc = self.nlp(clean_text)
        
        # Direct mutation of the SpaCy Doc for POS consistency
        for token in doc:
            if token.text.lower() in ["yesterday", "today", "tomorrow"] or (token.pos_ == "NOUN" and token.dep_ == "npadvmod"):
                token.pos = doc.vocab.strings["ADV"]
        
        # Stage 2
        spacy_parse, custom_parse = self.syntactic_processor.process(doc, clean_text)
        
        # Stage 3
        semantic_frames, wsd_data = self.semantic_analyzer.process(doc, clean_text)
        
        # Stage 4
        discourse_data = self.discourse_processor.process(doc, clean_text)
        
        return {
            "corrected_text": corrected_text,
            "spell_diff": spell_diff,
            "spacy_parse": spacy_parse,
            "custom_parse": custom_parse,
            "semantic_frames": semantic_frames,
            "wsd_data": wsd_data,
            "discourse_data": discourse_data
        }

def generate_summary(results):
    """Generates a summary table for the processed samples."""
    summary_data = []
    
    for filename, res in results.items():
        stats = {
            "File": os.path.basename(filename),
            "Spelling Corrections": len(res["spell_diff"]),
            "Entities Found": sum(len(frame["semantic_mentions"]) for frame in res["semantic_frames"]),
            "Coref Chains Resolved": len(res["discourse_data"]["coref_chains"]),
            "Discourse Relations Tagged": len(res["discourse_data"]["discourse_relations"])
        }
        summary_data.append(stats)
        
    df = pd.DataFrame(summary_data)
    return df

def main():
    pipeline = NLPPipeline()
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found.")
        return
        
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not files:
        print(f"No .txt files found in '{data_dir}'.")
        return
        
    results = {}
    print(f"Processing {len(files)} files...")
    
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Processing {os.path.basename(fpath)}...")
        results[fpath] = pipeline.process(text)
        
    df_summary = generate_summary(results)
    print("\n--- Pipeline Summary ---")
    print(df_summary.to_string(index=False))
    
    df_summary.to_csv("summary_table.csv", index=False)
    print("\nSummary saved to 'summary_table.csv'")

if __name__ == "__main__":
    main()

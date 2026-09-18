import re
from collections import Counter
import nltk
from nltk.util import ngrams

class SpellChecker:
    def __init__(self):
        # We'll use a basic dictionary and bigram model for context
        # For a real application, this should be loaded from a large corpus
        self.vocab = Counter()
        self.bigrams = Counter()
        self.total_words = 0
        
        # Load NLTK brown corpus for realistic word frequencies
        try:
            nltk.data.find('corpora/brown')
        except LookupError:
            nltk.download('brown')
            
        from nltk.corpus import brown
        for w in brown.words():
            if w.isalpha():
                self.vocab[w.lower()] += 1
            
        # Add some common words that might not be in the basic list
        common = ["there", "their", "they're", "its", "it's", "to", "too", "two", "your", "you're", "i", "a", "an", "the", "receive", "received", "message", "student", "assignment"]
        for w in common:
            self.vocab[w] += 1000
            
        # Simple bigram training data to demonstrate context awareness
        training_text = """
        they went there with their friends .
        it is too late to go to the store .
        two of the apples are yours .
        your book is on the table , you're going to read it .
        it's a good day for its warmth .
        i borrowed two books from the library .
        i need to borrow a book .
        we need to go .
        i want two of them .
        """
        tokens = re.findall(r'\w+', training_text.lower())
        self.vocab.update(tokens)
        self.bigrams.update(ngrams(tokens, 2))
        self.total_words = sum(self.vocab.values())
        
        # Initialize spaCy for NER / POS tagging to detect proper nouns
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            self.nlp = None

    def levenshtein_distance(self, s1, s2):
        """Calculates the Damerau-Levenshtein distance between two strings from scratch."""
        d = {}
        lenstr1 = len(s1)
        lenstr2 = len(s2)
        for i in range(-1, lenstr1+1):
            d[(i, -1)] = i + 1
        for j in range(-1, lenstr2+1):
            d[(-1, j)] = j + 1

        for i in range(lenstr1):
            for j in range(lenstr2):
                cost = 0 if s1[i] == s2[j] else 1
                
                d[(i, j)] = min(
                    d[(i-1, j)] + 1, # deletion
                    d[(i, j-1)] + 1, # insertion
                    d[(i-1, j-1)] + cost # substitution
                )
                if i > 0 and j > 0 and s1[i] == s2[j-1] and s1[i-1] == s2[j]:
                    d[(i, j)] = min(d[(i, j)], d[(i-2, j-2)] + cost) # transposition

        return d[(lenstr1-1, lenstr2-1)]

    def get_candidates(self, word, max_dist=2):
        """Returns candidate corrections with distance <= max_dist."""
        confused_words = {"there", "their", "they're", "its", "it's", "to", "too", "two", "your", "you're"}
        if word in self.vocab and word not in confused_words:
            return [(word, 0)] # Already correct
            
        candidates = []
        for w in self.vocab:
            # Optimization: length difference can't exceed max_dist
            if abs(len(w) - len(word)) <= max_dist:
                dist = self.levenshtein_distance(word, w)
                if dist <= max_dist:
                    candidates.append((w, dist))
                    
        # Sort by distance, then by frequency
        candidates.sort(key=lambda x: (x[1], -self.vocab[x[0]]))
        return candidates[:20] # Keep top 20

    def context_aware_correction(self, word, prev_word, next_word):
        """Uses n-gram probabilities to pick the best candidate."""
        candidates = self.get_candidates(word)
        
        if not candidates:
            return word, "No correction found", 0.0
            
        if len(candidates) == 1 and candidates[0][1] == 0:
            return word, "Correct as is", 1.0

        best_candidate = candidates[0][0]
        best_score = -1.0
        reason = "Closest edit distance"
        best_dist = candidates[0][1]

        for cand, dist in candidates:
            # Simple scoring: frequency + context bonus
            score = self.vocab.get(cand, 1) / self.total_words
            
            has_context = False
            # Bigram context probability
            if prev_word:
                bg_count = self.bigrams.get((prev_word, cand), 0)
                if bg_count > 0:
                    score *= (bg_count * 100000) # Big bonus for context match
                    reason = f"N-gram probability with '{prev_word}'"
                    has_context = True
                    
            if next_word:
                bg_count = self.bigrams.get((cand, next_word), 0)
                if bg_count > 0:
                    score *= (bg_count * 100000)
                    if not has_context:
                        reason = f"N-gram probability with '{next_word}'"
                    has_context = True

            # Strong penalty for edit distance
            score /= (10 ** dist)

            if score > best_score:
                best_score = score
                best_candidate = cand
                best_dist = dist

        # Confidence check: Do not use edit distance alone for unknown/low-frequency candidates without context
        if best_dist >= 2 and "N-gram" not in reason:
            if len(word) <= 4 or best_score < 1e-6:
                return word, "Low confidence, leaving unchanged", best_score
                
        # Additional safety for arbitrary unknown words being forced to a distance 1 candidate
        if best_dist == 1 and "N-gram" not in reason and self.vocab.get(best_candidate, 0) < 5:
            return word, "Low confidence frequency, leaving unchanged", best_score

        # Also, do not change word if the candidate frequency is very low and it's a 2 edit distance
        if best_dist >= 2 and self.vocab.get(best_candidate, 0) < 50 and "N-gram" not in reason:
             return word, "Low confidence frequency (dist 2), leaving unchanged", best_score

        return best_candidate, f"Distance: {best_dist}, Reason: {reason}", best_score

    def process(self, text):
        """Processes text and returns corrected text and diffs."""
        
        # 1. Use spaCy to find protected proper nouns
        protected_words = set()
        if self.nlp:
            doc = self.nlp(text)
            for token in doc:
                # Protect if it's a proper noun or an entity
                if token.pos_ == "PROPN" or token.ent_type_ in ["PERSON", "ORG", "GPE", "LOC", "FAC"]:
                    protected_words.add(token.text.lower())

        original_tokens = re.findall(r"[\w']+|[^\w']+", text) # Preserve ALL case/whitespace/punctuation exactly
        
        # We need a parallel list to replace words
        words_only = [(i, t.lower()) for i, t in enumerate(original_tokens) if re.match(r"^[\w']+$", t)]
        
        corrections = []
        corrected_text_tokens = list(original_tokens)
        
        for idx_in_words, (idx_in_tokens, word) in enumerate(words_only):
            orig_token = original_tokens[idx_in_tokens]
            
            # PROTECT PROPER NOUNS / ENTITIES
            is_protected = False
            if word in protected_words:
                is_protected = True
            
            # Additional heuristic: Protect capitalized words not at start of sentence
            if orig_token.istitle() and idx_in_words > 0:
                # Check previous non-whitespace tokens to see if it's start of sentence
                prev_text = "".join(original_tokens[:idx_in_tokens]).strip()
                if not (prev_text.endswith('.') or prev_text.endswith('?') or prev_text.endswith('!')):
                    is_protected = True
                    
            if is_protected:
                continue

            prev_word = words_only[idx_in_words-1][1] if idx_in_words > 0 else None
            next_word = words_only[idx_in_words+1][1] if idx_in_words < len(words_only)-1 else None
            
            corrected_word, reason, score = self.context_aware_correction(word, prev_word, next_word)
            
            if corrected_word != word and "Low confidence" not in reason:
                # Try to preserve original casing
                if orig_token.istitle():
                    corrected_word = corrected_word.capitalize()
                elif orig_token.isupper():
                    corrected_word = corrected_word.upper()
                    
                if corrected_word != orig_token and corrected_word.lower() != orig_token.lower():
                    corrected_text_tokens[idx_in_tokens] = corrected_word
                    corrections.append({
                        'original': orig_token,
                        'corrected': corrected_word,
                        'reason': reason
                    })
                
        corrected_text = "".join(corrected_text_tokens)
        return corrected_text, corrections

if __name__ == "__main__":
    checker = SpellChecker()
    test_text = "Rahul sent the report to Priya yestarday."
    corrected, diff = checker.process(test_text)
    print("Original:", test_text)
    print("Corrected:", corrected)
    print("Diffs:", diff)

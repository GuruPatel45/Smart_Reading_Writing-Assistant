import spacy
from spacy import displacy
import nltk

class SyntacticProcessor:
    def __init__(self):
        # Define a small custom grammar for the CYK/Recursive Descent parser
            
        # Define a small custom grammar for the CYK/Recursive Descent parser
        # Expanded to support demo sentences
        self.grammar_string = """
            S -> S_Clause Punct | S_Clause Punct Conj S_Clause Punct | Aux NP VP Punct
            S_Clause -> NP VP | Discourse Punct NP VP | NP VP InfClause
            NP -> Det N | Pronoun | ProperNoun | Det N PP | Pronoun PP | ProperNoun PP
            VP -> V NP | V NP PP | V | V Adj | V NP Adv | V NP PP Adv | Aux Neg V NP | V PP | V InfClause
            InfClause -> Part V NP | Part V
            PP -> P NP
            Det -> 'the' | 'a' | 'an' | 'my' | 'his' | 'her'
            N -> 'dog' | 'cat' | 'man' | 'telescope' | 'park' | 'student' | 'file' | 'email' | 'book' | 'report' | 'bank' | 'money'
            V -> 'saw' | 'ate' | 'walked' | 'read' | 'send' | 'is' | 'sent' | 'understand' | 'opened' | 'went' | 'deposit'
            Aux -> 'could' | 'can' | 'will' | 'would'
            Neg -> 'not'
            P -> 'in' | 'on' | 'by' | 'with' | 'to'
            Part -> 'to'
            Pronoun -> 'he' | 'she' | 'it' | 'i' | 'you' | 'me'
            ProperNoun -> 'john' | 'mary' | 'bob' | 'rahul' | 'priya' | 'amit' | 'aarav' | 'meera'
            Adj -> 'good' | 'bad' | 'tall' | 'short'
            Adv -> 'yesterday' | 'today' | 'tomorrow' | 'again'
            Discourse -> 'however' | 'therefore' | 'although' | 'thus'
            Conj -> 'but' | 'and' | 'or'
            Punct -> '.' | ',' | '?' | '!'
        """
        # Parse grammar into rules
        self.rules = []
        for line in self.grammar_string.split('\n'):
            line = line.strip()
            if not line: continue
            lhs, rhs = line.split('->')
            lhs = lhs.strip()
            for prod in rhs.split('|'):
                prod = [p.strip().strip("'") for p in prod.split()]
                self.rules.append((lhs, prod))

    def spacy_parse(self, doc):
        """Extracts POS tags and dependency parse from the existing document."""
        sentences = []
        for sent in doc.sents:
            tokens = []
            for token in sent:
                # The Token POS is already mutated globally in pipeline.py
                pos = token.pos_
                    
                tokens.append({
                    'text': token.text,
                    'pos': pos,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'head': token.head.text
                })
            sentences.append({
                'text': sent.text,
                'tokens': tokens,
                # Explicitly passing the individual sentence doc subset to displacy prevents multi-sentence rendering bugs
                'displacy_html': displacy.render(sent, style='dep', jupyter=False, page=False)
            })
        return sentences

    def tokenize_for_cfg(self, text):
        import re
        # Preserve punctuation natively
        tokens = re.findall(r"[\w']+|[^\w\s]+", text.lower())
        return tokens

    def recursive_descent_parse(self, tokens, start_symbol='S'):
        """
        A simple manual implementation of a top-down recursive descent parser.
        Returns the first valid parse tree found, or None.
        """
        def parse(symbol, index):
            # If symbol is a terminal (a string not in LHS of any rule)
            if not any(lhs == symbol for lhs, rhs in self.rules):
                if index < len(tokens) and tokens[index] == symbol:
                    return symbol, index + 1
                return None, index

            # If symbol is non-terminal, try all its productions
            for lhs, rhs in self.rules:
                if lhs == symbol:
                    current_index = index
                    children = []
                    valid = True
                    for sym in rhs:
                        child_tree, next_index = parse(sym, current_index)
                        if child_tree is None:
                            valid = False
                            break
                        children.append(child_tree)
                        current_index = next_index
                        
                    if valid:
                        return {symbol: children}, current_index
            return None, index

        tree, final_index = parse(start_symbol, 0)
        if tree is not None and final_index == len(tokens):
            return tree
        return None
        
    def process(self, doc, text):
        spacy_results = self.spacy_parse(doc)
        
        # Try custom parser on sentences
        custom_parse_results = []
        for sent in doc.sents:
            tokens = self.tokenize_for_cfg(sent.text)
            tree = self.recursive_descent_parse(tokens)
            custom_parse_results.append({
                'text': sent.text,
                'tokens': tokens,
                'tree': tree
            })
            
        return spacy_results, custom_parse_results

if __name__ == "__main__":
    processor = SyntacticProcessor()
    spacy_res, custom_res = processor.process("Rahul sent the report to Priya yesterday.")
    
    print("SpaCy Result:")
    for token in spacy_res[0]['tokens']:
        print(f"{token['text']} ({token['pos']}) <--{token['dep']}-- {token['head']}")
        
    print("\nCustom Parse Tree:")
    import json
    print(json.dumps(custom_res[0]['tree'], indent=2))

import spacy
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

class SemanticAnalyzer:
    def __init__(self):
        # Ensure wordnet is downloaded

        # Ensure wordnet is downloaded
        try:
            wn.synsets('dog')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('punkt')
            
    def extract_roles_and_entities(self, doc):
        """
        Uses spaCy dependency parsing to extract rough semantic roles (agent, action, patient, recipient)
        and distinguishes between Named Entities and Semantic Mentions.
        """
        sentences_data = []
        for sent in doc.sents:
            sent_data = {
                "text": sent.text,
                "agent": None,
                "action": None,
                "patient": None,
                "recipient": None,
                "entities": [{"text": ent.text, "label": ent.label_} for ent in sent.ents],
                "semantic_mentions": []
            }
            
            # Extract semantic mentions (meaningful entities in discourse, excluding pronouns/temporals)
            mentions = []
            for chunk in doc.noun_chunks:
                if chunk.root.sent == sent:
                    root_text = chunk.root.text
                    # Filter out pronouns and temporal adverbs pretending to be nouns
                    if chunk.root.pos_ not in ["PRON"] and root_text.lower() not in ["yesterday", "today", "tomorrow"]:
                        if root_text not in mentions:
                            mentions.append(root_text)
            sent_data["semantic_mentions"] = mentions
            
            # Proper noun candidates
            propns = []
            for token in sent:
                if token.pos_ == "PROPN" and token.text not in propns:
                    propns.append(token.text)
            sent_data["proper_noun_candidates"] = propns
            
            # Find all meaningful predicates (verbs) in the sentence
            clauses = []
            for token in sent:
                if token.pos_ in ["VERB", "AUX"] and token.dep_ in ["ROOT", "conj", "advcl", "xcomp"]:
                    clause = {
                        "action": token.text,
                        "agent": None,
                        "patient": None,
                        "recipient": None,
                        "destination": None,
                        "negation": None,
                        "modal": None
                    }
                    
                    for child in token.children:
                        if "subj" in child.dep_:
                            clause["agent"] = " ".join([t.text for t in child.subtree])
                        elif "obj" in child.dep_:
                            clause["patient"] = " ".join([t.text for t in child.subtree])
                        elif child.dep_ == "neg":
                            clause["negation"] = child.text
                        elif child.dep_ == "aux" and child.pos_ == "AUX":
                            if child.text.lower() != "to":
                                clause["modal"] = child.text
                        elif "dative" in child.dep_:
                            clause["recipient"] = " ".join([t.text for t in child.subtree])
                        elif "prep" in child.dep_ and child.text.lower() in ["to", "for"]:
                            for grandchild in child.children:
                                if "pobj" in grandchild.dep_:
                                    is_person = grandchild.ent_type_ == "PERSON" or grandchild.pos_ in ["PROPN", "PRON"]
                                    if is_person:
                                        clause["recipient"] = " ".join([t.text for t in grandchild.subtree])
                                    else:
                                        clause["destination"] = " ".join([t.text for t in grandchild.subtree])
                                        
                    # Infer implicit agents for coordinated/adverbial verbs
                    if clause["agent"] is None and token.dep_ in ["conj", "advcl", "xcomp"]:
                        head = token.head
                        for child in head.children:
                            if "subj" in child.dep_:
                                clause["agent"] = " ".join([t.text for t in child.subtree])
                                break
                                
                    clauses.append(clause)
            
            sent_data["clauses"] = clauses
            
            sentences_data.append(sent_data)
        return sentences_data

    def get_word_overlap(self, set1, set2):
        return len(set1.intersection(set2))

    def lesk_algorithm(self, word, context_sentence):
        """
        Implementation of the Simplified Lesk algorithm from scratch.
        word: The ambiguous word to disambiguate
        context_sentence: The sentence in which the word appears
        """
        synsets = wn.synsets(word, pos=wn.NOUN)
        if not synsets:
            return None, "No synsets found"

        best_sense = synsets[0]
        max_overlap = 0
        
        context_words = set(word_tokenize(context_sentence.lower()))
        context_words.discard(word.lower())
        
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            context_words = {w for w in context_words if w not in stop_words}
        except:
            pass

        for sense in synsets:
            signature = set(word_tokenize(sense.definition().lower()))
            for example in sense.examples():
                signature.update(word_tokenize(example.lower()))
                
            overlap = self.get_word_overlap(signature, context_words)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense

        return best_sense.name(), best_sense.definition()

    def disambiguate_ambiguous_words(self, doc):
        """Disambiguates words dynamically if they have multiple synsets."""
        results = []
        for sent in doc.sents:
            for token in sent:
                # Target nouns that are reasonably long and have multiple meanings
                if token.pos_ == "NOUN" and len(token.text) >= 3:
                    synsets = wn.synsets(token.lemma_, pos=wn.NOUN)
                    if len(synsets) >= 3: # Highly ambiguous
                        sense, definition = self.lesk_algorithm(token.text, sent.text)
                        if sense:
                            results.append({
                                "word": token.text,
                                "sentence": sent.text,
                                "sense_id": sense,
                                "definition": definition
                            })
        return results

    def process(self, doc, text):
        srl_ner_data = self.extract_roles_and_entities(doc)
        wsd_data = self.disambiguate_ambiguous_words(doc)
        
        return srl_ner_data, wsd_data

if __name__ == "__main__":
    analyzer = SemanticAnalyzer()
    text = "Rahul sent the report to Priya yesterday. She read it. However, she could not understand the report. Can you send me the file?"
    srl, wsd = analyzer.process(text)
    
    import json
    print("Semantic Roles & Entities:")
    print(json.dumps(srl, indent=2))
    
    print("\nWord Sense Disambiguation:")
    print(json.dumps(wsd, indent=2))

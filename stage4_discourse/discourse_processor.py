import spacy
import re

class DiscourseProcessor:
    def __init__(self):
        # Extended discourse connectives dictionary (Bug 9)

        # Extended discourse connectives dictionary (Bug 9)
        self.connectives = {
            "contrast": ["however", "although", "but", "nevertheless", "on the other hand", "yet"],
            "cause": ["because", "therefore", "thus", "consequently", "since", "as a result", "so"],
            "elaboration": ["furthermore", "moreover", "in addition", "also", "for example", "specifically"],
            "sequence": ["then", "next", "finally", "subsequently"],
            "concession": ["even though", "despite", "in spite of"]
        }

    def heuristic_coreference_resolution(self, doc):
        """
        A practical rule-based coreference resolver.
        Matches 3rd person pronouns to the nearest preceding noun phrase 
        that broadly matches in gender/animacy based on simplistic rules.
        """
        chains = []
        recent_entities = [] # list of (entity_text, is_person, is_plural)

        for sent in doc.sents:
            for token in sent:
                # Add to recent entities if it's a noun
                if token.pos_ in ["NOUN", "PROPN"]:
                    
                    # BUG 8 Fix: Do not register temporal adverbs masquerading as nouns as candidates for "it"
                    if token.text.lower() in ["yesterday", "today", "tomorrow"]:
                        continue
                        
                    is_person = token.ent_type_ == "PERSON" or token.text.lower() in ["man", "woman", "boy", "girl", "john", "mary", "rahul", "priya", "amit"]
                    is_plural = token.tag_ in ["NNS", "NNPS"]
                    
                    # Get full NP
                    np_text = token.text
                    for chunk in doc.noun_chunks:
                        if token in chunk:
                            np_text = chunk.text
                            break
                    recent_entities.append((np_text, is_person, is_plural))
                
                # Resolve pronouns
                if token.pos_ == "PRON":
                    pronoun = token.text.lower()
                    target = None
                    
                    if pronoun in ["he", "him", "his", "she", "her", "hers"]:
                        # Look for nearest person, singular
                        for ent, is_p, is_pl in reversed(recent_entities):
                            if is_p and not is_pl:
                                target = ent
                                break
                    elif pronoun in ["it", "its"]:
                        # Look for nearest non-person, singular
                        for ent, is_p, is_pl in reversed(recent_entities):
                            if not is_p and not is_pl:
                                target = ent
                                break
                    elif pronoun in ["they", "them", "their", "theirs"]:
                        # Look for nearest plural
                        for ent, is_p, is_pl in reversed(recent_entities):
                            if is_pl:
                                target = ent
                                break
                    
                    if target:
                        chains.append({"pronoun": pronoun, "antecedent": target, "sentence": sent.text})

        # Group by antecedent (Bug 4)
        grouped_chains = {}
        for chain in chains:
            ant = chain["antecedent"]
            if ant not in grouped_chains:
                grouped_chains[ant] = {"antecedent": ant, "mentions": []}
            grouped_chains[ant]["mentions"].append({"pronoun": chain["pronoun"], "sentence": chain["sentence"]})

        return list(grouped_chains.values())

    def detect_discourse_relations(self, text):
        """Detects discourse connectives dynamically based on rules."""
        relations = []
        text_lower = text.lower()
        
        for relation_type, keywords in self.connectives.items():
            for kw in keywords:
                # Use regex to find whole words
                pattern = r'\b' + re.escape(kw) + r'\b'
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    relations.append({
                        "connective": kw,
                        "relation_type": relation_type,
                        "index": match.start()
                    })
                    
        # Sort by appearance in text
        relations.sort(key=lambda x: x["index"])
        return relations

    def apply_pragmatic_inference(self, doc):
        """
        Rule-based pragmatic inference detector.
        Detects pragmatic inferences, such as indirect requests.
        E.g., "Could you...", "Can you...", "Would you please..." usually mean an imperative command.
        """
        inferences = []
        
        for sent in doc.sents:
            text = sent.text.lower().strip()
            # Extended pattern matching for indirect requests (Bug 10)
            if text.startswith("could you ") or text.startswith("can you ") or text.startswith("would you "):
                action_part = text
                
                # Strip out the modal and polite markers to get the core intended action
                for prefix in ["could you please ", "can you please ", "would you please ", "could you ", "can you ", "would you "]:
                    if action_part.startswith(prefix):
                        action_part = action_part[len(prefix):]
                        break
                        
                action_part = action_part.replace("?", "").strip()
                
                inferences.append({
                    "original_text": sent.text,
                    "type": "indirect_request",
                    "inferred_meaning": f"Imperative: please {action_part}"
                })
                
        return inferences

    def process(self, doc, text):
        
        coref_chains = self.heuristic_coreference_resolution(doc)
        discourse_rels = self.detect_discourse_relations(text)
        pragmatics = self.apply_pragmatic_inference(doc)
        
        return {
            "coref_chains": coref_chains,
            "discourse_relations": discourse_rels,
            "pragmatics": pragmatics
        }

if __name__ == "__main__":
    processor = DiscourseProcessor()
    text = "Rahul sent the report to Priya yesterday. She read it. However, she could not understand the report. Can you send me the file?"
    results = processor.process(text)
    
    import json
    print(json.dumps(results, indent=2))

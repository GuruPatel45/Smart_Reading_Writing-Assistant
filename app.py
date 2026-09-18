import streamlit as st
import time
import json
from pipeline import NLPPipeline

# Initialize the pipeline
@st.cache_resource
def load_pipeline():
    return NLPPipeline()

st.set_page_config(layout="wide", page_title="Smart Reading & Writing Assistant")

pipeline = load_pipeline()

st.title("🧠 Smart Reading & Writing Assistant")
st.markdown("An end-to-end NLP pipeline analyzing Spelling, Syntax, Semantics, and Discourse.")

# Text input
text_input = st.text_area("Enter your text here...", height=150, value="Rahul sent the report to Priya yestarday.\nShe read it.\nHowever, she could not understand the report.\nCan you send me the file?")

if st.button("Analyze Text"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Running NLP Pipeline..."):
            start_time = time.time()
            try:
                results = pipeline.process(text_input)
                latency = time.time() - start_time
                st.success(f"Analysis complete in {latency:.3f} seconds!")
                
                # --- STAGE 1: SPELL CHECKING ---
                st.markdown("---")
                st.header("1. SPELL CHECKING")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Text")
                    st.write(text_input)
                with col2:
                    st.subheader("Corrected Text")
                    st.write(results["corrected_text"])
                
                st.subheader("Corrections Made")
                if results["spell_diff"]:
                    for diff in results["spell_diff"]:
                        st.info(f"**{diff['original']}** ➔ **{diff['corrected']}** *(Reason: {diff['reason']})*")
                else:
                    st.success("No spelling errors detected!")
                
                # --- STAGE 2: SYNTACTIC ANALYSIS ---
                st.markdown("---")
                st.header("2. SYNTACTIC ANALYSIS")
                
                tabs_syn = st.tabs(["POS & Dependencies", "Parse Tree", "Custom CFG Parser"])
                
                with tabs_syn[0]:
                    for i, sent_data in enumerate(results["spacy_parse"]):
                        st.markdown(f"**Sentence {i+1}:** *{sent_data['text']}*")
                        # Format as a table
                        pos_data = [{"Token": t['text'], "POS": t['pos'], "Tag": t['tag'], "Dependency": t['dep'], "Head": t['head']} for t in sent_data['tokens']]
                        st.table(pos_data)
                
                with tabs_syn[1]:
                    for i, sent_data in enumerate(results["spacy_parse"]):
                        st.markdown(f"**Sentence {i+1}:**")
                        st.components.v1.html(sent_data['displacy_html'], scrolling=True, height=400)
                
                with tabs_syn[2]:
                    st.markdown("*A simple top-down recursive descent parser demonstrating parsing mechanics on a small grammar.*")
                    for i, custom_data in enumerate(results["custom_parse"]):
                        st.markdown(f"**Sentence {i+1}:** *{custom_data['text']}*")
                        if custom_data['tree']:
                            st.code(json.dumps(custom_data['tree'], indent=2), language='json')
                        else:
                            st.warning("No valid CFG parse found for this sentence with the limited grammar.")
                
                # --- STAGE 3: SEMANTIC ANALYSIS ---
                st.markdown("---")
                st.header("3. SEMANTIC ANALYSIS")
                
                tabs_sem = st.tabs(["Semantic Roles & Entities", "Word Sense Disambiguation", "JSON Output"])
                
                with tabs_sem[0]:
                    for i, frame in enumerate(results["semantic_frames"]):
                        st.markdown(f"**Sentence {i+1}:** *{frame['text']}*")
                        
                        for c_idx, clause in enumerate(frame['clauses']):
                            st.markdown(f"**Clause {c_idx+1} Predicate: `{clause['action']}`**")
                            cols = st.columns(6)
                            cols[0].metric("Agent", clause['agent'] or "N/A")
                            cols[1].metric("Action", clause['action'] or "N/A")
                            cols[2].metric("Patient", clause['patient'] or "N/A")
                            cols[3].metric("Recipient", clause['recipient'] or "N/A")
                            cols[4].metric("Destination", clause['destination'] or "N/A")
                            mod_str = ""
                            if clause['negation']: mod_str += f"NEG: {clause['negation']} "
                            if clause['modal']: mod_str += f"MOD: {clause['modal']}"
                            cols[5].metric("Modifiers", mod_str.strip() or "N/A")
                        
                        col_e1, col_e2, col_e3 = st.columns(3)
                        with col_e1:
                            st.markdown("**Named Entities (NER):**")
                            if frame['entities']:
                                ent_str = ", ".join([f"{e['text']} ({e['label']})" for e in frame['entities']])
                                st.info(ent_str)
                            else:
                                st.markdown("*None*")
                        with col_e2:
                            st.markdown("**Proper-Noun Candidates:**")
                            if frame.get('proper_noun_candidates'):
                                st.info(", ".join(frame['proper_noun_candidates']))
                            else:
                                st.markdown("*None*")
                        with col_e3:
                            st.markdown("**Semantic Mentions:**")
                            if frame['semantic_mentions']:
                                st.info(", ".join(frame['semantic_mentions']))
                            else:
                                st.markdown("*None*")
                            
                with tabs_sem[1]:
                    if results["wsd_data"]:
                        for wsd in results["wsd_data"]:
                            st.success(f"**{wsd['word']}** (in *'{wsd['sentence']}'*) ➔ **{wsd['sense_id']}**: {wsd['definition']}")
                    else:
                        st.info("No heavily ambiguous target words detected for WSD.")
                        
                with tabs_sem[2]:
                    st.code(json.dumps(results["semantic_frames"], indent=2), language='json')
                
                # --- STAGE 4: DISCOURSE & PRAGMATIC ANALYSIS ---
                st.markdown("---")
                st.header("4. DISCOURSE & PRAGMATIC ANALYSIS")
                
                d_data = results["discourse_data"]
                
                st.subheader("Coreference Chains")
                if d_data["coref_chains"]:
                    for chain in d_data["coref_chains"]:
                        mentions_str = ", ".join([f"{m['pronoun']} (in: '{m['sentence']}')" for m in chain['mentions']])
                        st.info(f"**{chain['antecedent']}** ➔ {mentions_str}")
                else:
                    st.write("*No coreferences resolved.*")
                    
                st.subheader("Discourse Relations")
                if d_data["discourse_relations"]:
                    for rel in d_data["discourse_relations"]:
                        st.info(f"**{rel['connective'].capitalize()}** ➔ **{rel['relation_type'].capitalize()}**")
                else:
                    st.write("*No discourse connectors detected.*")
                    
                st.subheader("Pragmatic Inference")
                if d_data["pragmatics"]:
                    for prag in d_data["pragmatics"]:
                        st.success(f"**{prag['original_text']}**  \n➔ **{prag['type'].replace('_', ' ').capitalize()}**  \n➔ Intended action: {prag['inferred_meaning']}")
                else:
                    st.write("*No pragmatic inferences detected.*")
                    
                # --- STAGE 5: FINAL SUMMARY ---
                st.markdown("---")
                st.header("5. FINAL SUMMARY")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Total Sentences", len(results["spacy_parse"]))
                col_s1.metric("Spelling Corrections", len(results["spell_diff"]))
                
                total_semantic_entities = sum(len(f['semantic_mentions']) for f in results["semantic_frames"])
                col_s2.metric("Total Semantic Mentions", total_semantic_entities)
                col_s2.metric("Coreference Chains", len(d_data["coref_chains"]))
                
                col_s3.metric("Discourse Relations", len(d_data["discourse_relations"]))
                col_s3.metric("Pragmatic Inferences", len(d_data["pragmatics"]))

            except Exception as e:
                import traceback
                st.error(f"An error occurred during analysis: {e}\n\n{traceback.format_exc()}")

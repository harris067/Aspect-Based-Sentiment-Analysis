import spacy
import pandas as pd
import json

nlp = spacy.load("en_core_web_trf")


def get_head_noun(token):
    """
    For a given noun token, return the head of its compound (if any).
    E.g., for 'customer service', return 'service'.
    """
    if token.dep_ == "compound" and token.head.pos_ in {"NOUN", "PROPN"}:
        return get_head_noun(token.head)
    return token.text

def clean_opinion(token):
    """
    Return the text for an opinion token.
    Hyphenated adjectives (e.g., 'pre-prepared', 'straight-forward') are kept intact.
    """
    return token.text.strip()

def resolve_pronoun(token, last_noun):
    """
    If the token is a demonstrative pronoun ('it', 'this', 'that', 'they'),
    replace it with the most recently encountered noun (last_noun).
    Otherwise, return the token's text.
    """
    if token.lower_ in {"it", "this", "that", "they"} and last_noun:
        return last_noun
    return token.text

def extract_aspect_opinion_pairs(sentence):
    """
    Extract aspect-opinion pairs from the sentence using multiple patterns:
    
    Pattern A: Direct adjectival modifier (amod).
      - E.g., "coffee is OUTSTANDING" -> (coffee, OUTSTANDING)
    
    Pattern B: Copular constructions (acomp, attr).
      - E.g., "the service is not good" -> (service, not good)
    
    Pattern C: Verbal constructions with negation.
      - E.g., "the food wasn't cooked fresh" -> (food, not cooked)
        Here, if a negation is detected on a verb, capture only the negation word (converted
        if necessary) and the immediately following token.
    
    Pattern D: Loose adjectives in descriptive clauses.
      - E.g., "Straight-forward in presentation" -> (presentation, Straight-forward)
    
    For pronoun aspects, the immediately preceding noun (last encountered noun) is used.
    """
    doc = nlp(sentence)
    pairs = []
    last_noun = None  

    
    for sent in doc.sents:
        
        for token in sent:
            if token.pos_ in {"NOUN", "PROPN"}:
                last_noun = get_head_noun(token)

        
        for token in sent:
            if token.pos_ == "ADJ" and token.dep_ == "amod":
                if token.head.pos_ in {"NOUN", "PROPN"}:
                    aspect = get_head_noun(token.head)
                    
                    if token.text.lower() not in {"asian", "japanese", "chinese"}:
                        opinion = clean_opinion(token)
                        pairs.append((aspect, opinion))
                        last_noun = aspect

        
        for token in sent:
            if token.pos_ == "ADJ" and token.dep_ in {"acomp", "attr"}:
                subj = None
                for child in token.head.children:
                    if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                        subj = child
                        break
                if subj:
                    aspect = (get_head_noun(subj)
                              if subj.pos_ != "PRON"
                              else resolve_pronoun(subj, last_noun))
                    neg = ""
                    for child in token.head.children:
                        if child.dep_ == "neg":
                            neg = child.text + " "
                            
                            if neg.strip() == "n't":
                                neg = "not "
                            break
                    opinion = neg + clean_opinion(token)
                    pairs.append((aspect, opinion))
                    last_noun = aspect

        
        for token in sent:
            if token.pos_ == "VERB" and token.tag_ != "AUX":
                neg_token = None
                for child in token.children:
                    if child.dep_ == "neg":
                        neg_token = child
                        break
                if neg_token:
                    subj = None
                    for child in token.children:
                        if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                            subj = child
                            break
                    if subj:
                        aspect = (get_head_noun(subj)
                                  if subj.pos_ != "PRON"
                                  else resolve_pronoun(subj, last_noun))
                        
                        subtree_tokens = list(token.subtree)
                        subtree_tokens = sorted(subtree_tokens, key=lambda t: t.i)
                        try:
                            neg_index = subtree_tokens.index(neg_token)
                        except ValueError:
                            neg_index = 0
                    
                        selected_tokens = subtree_tokens[neg_index: neg_index+2]
                        
                        opinion_tokens = []
                        for t in selected_tokens:
                            if t.text.strip() == "n't":
                                opinion_tokens.append("not")
                            else:
                                opinion_tokens.append(t.text)
                        opinion = " ".join(opinion_tokens)
                        pairs.append((aspect, opinion))
                        last_noun = aspect

        
        for token in sent:
            if token.pos_ == "ADJ" and token.dep_ not in {"amod", "acomp", "attr"}:
                for i in range(token.i - 1, -1, -1):
                    if doc[i].pos_ in {"NOUN", "PROPN"}:
                        aspect = get_head_noun(doc[i])
                        opinion = clean_opinion(token)
                        pairs.append((aspect, opinion))
                        last_noun = aspect
                        break

    
    unique_pairs = list(set(pairs))
    return unique_pairs

def main():
    input_file = "merged.csv" #From existing work  
    output_file = "dependancy_output.csv" 

    df = pd.read_csv(input_file, encoding="utf-8")
    if "Sentence" in df.columns and "sentence" not in df.columns:
        df.rename(columns={"Sentence": "sentence"}, inplace=True)

    output_data = []
    for _, row in df.iterrows():
        id=row["id"]
        sentence = row["sentence"]
        pairs = extract_aspect_opinion_pairs(sentence)
        output_data.append({
            "id":id,
            "sentence": sentence,
            "aspect_opinion_pairs": json.dumps(pairs, ensure_ascii=False)
        })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Extraction complete. Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()


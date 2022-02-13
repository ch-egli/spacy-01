import spacy
import pandas as pd

# nlp = spacy.load("de_core_news_sm")
nlp = spacy.load("de_dep_news_trf")
document = nlp(
    "Der 24-jährige Nidwaldner wird seiner Favoritenrolle gerecht und holt im Riesenslalom die ersehnte Goldmedaille. "
    "In Yanqing war es dunkler und dunkler geworden, als Marco Odermatt um ca. 16:00 Uhr Ortszeit zu Olympia-Gold "
    "startete. Der Nidwaldner behielt den Durchblick und vor allem die Nerven und rettete 19 Hundertstel ins Ziel. An "
    "seinen ersten Spielen sicherte er sich sein erstes Edelmetall, es wurde die ebenso ersehnte wie verdiente "
    "Goldmedaille.")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def extract_tokens_plus_meta(doc: spacy.tokens.doc.Doc):
    """Extract tokens and metadata from individual spaCy doc."""
    data = []
    for sentence_counter, sentence in enumerate(doc.sents):
        for token in sentence:
            data.insert(0, [sentence_counter, token.text, token.lemma_, token.pos_, token.morph,
                            token.is_alpha, token.is_digit, token.is_punct, token.is_sent_start, token.is_sent_end])
    # return array in reverse order
    return data[::-1]


def tidy_tokens(doc):
    """Extract tokens and metadata from list of spaCy docs."""

    cols = [
        "doc_id", "sentence_id", "token", "lemma",
        "pos (Wortart)", "morph", "is_alpha", "is_digit", "is_punct", "sentence-start", "sentence-end"
    ]

    meta_df = []
    meta = extract_tokens_plus_meta(doc)
    meta = pd.DataFrame(meta)
    meta.columns = cols[1:]
    meta_df.append(meta)

    return pd.concat(meta_df)


print(tidy_tokens(document))

# https://universaldependencies.org/u/pos/
# https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/mitarbeiter-innen/hagen/STTS_Tagset_Tiger
# https://files.ifi.uzh.ch/cl/siclemat/lehre/hs13/ecl1/script/script.pdf

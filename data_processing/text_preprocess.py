import spacy

nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("sentencizer")


def sentence_segmentation(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # continue splitting by \n
    final_sentences = []
    for sent in sentences:
        final_sentences.extend([s.strip() for s in sent.split("\n") if s.strip()])
    sentences = final_sentences
    return sentences

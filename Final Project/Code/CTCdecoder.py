import os
from dataclasses import dataclass

import torch
import torchaudio
from torchaudio.models.decoder._ctc_decoder import CTCDecoderLM, CTCDecoderLMState

import LSTMCorpus, PreProcessing

from torchaudio.models.decoder import ctc_decoder

LM_WEIGHT = 0
WORD_SCORE = -2


def build_tokens():
    """
    builds the tokens list from the PreProcessing.py file
    """
    tokens_string = PreProcessing.VOCABULARY
    tokens = [token for token in tokens_string]
    print("Building tokens list: ", tokens)

    # turn the tokens list into a dictionary
    tokens_d = {token: i for i, token in enumerate(tokens)}
    return tokens, tokens_d


def build_lexicon(data_path='an4/train/an4/'):
    corpus = LSTMCorpus.collect_corpus(txt_files=LSTMCorpus.get_corpus_paths(data_path), data_path=data_path)
    lexicon = set()
    for phrase in corpus:
        for word in phrase.split():
            lexicon.add(word.lower())
    lexicon = sorted(list(lexicon))
    with open('lexicon.txt', 'w') as f:
        for word in lexicon:
            spelled_word = ' '.join(list(word))
            f.write(word + ' ' + spelled_word + ' \n')
        # f.write('<unk>  \n')

    lexicon.append('<unk>')
    # make the lexicon into a dictionary
    lexicon = {word: i for i, word in enumerate(lexicon)}
    return lexicon


class Custom_LM(CTCDecoderLM):
    def __init__(self, language_model: LSTMCorpus.LSTMModel, lexicon):
        CTCDecoderLM.__init__(self)
        self.language_model = language_model
        self.sil = 1  # index for silent token in the language model
        self.states = {}
        self.vocab = language_model.vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.lexicon = lexicon
        self.reverse_lexicon = {v: k for k, v in lexicon.items()}
        language_model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        with torch.no_grad():
            score = self.language_model.score_sequence(torch.tensor([self.sil]))

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, usr_token_idx: int):
        outstate = state.child(usr_token_idx)
        if outstate not in self.states:
            seq = torch.tensor([self.vocab[char] for char in self.reverse_lexicon[
                usr_token_idx]]) if usr_token_idx != 98 else None # convert the token index (which is
            # actually
            # word idx)
            # to the
            # character index. then dont unsqueeze to add the batch dimension, it's already there
            score = self.language_model.score_sequence(seq) if seq is not None else 0
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.sil)


def load_model(lex, embedding_dim=100, hidden_dim=128):
    model = LSTMCorpus.LSTMModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load("lstm_model.pth"))
    return Custom_LM(model, lex)


@dataclass
class Files:
    lex = build_lexicon()
    lexicon = 'lexicon.txt'
    tokens, tokens_d = build_tokens()
    lm = load_model(lex)
    # lm = None

def create_beam_search_decoder(files: Files):
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        lm_dict='lm_dict.txt',
        nbest=1,
        beam_size=20,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
        blank_token='?',
        sil_token=' ',
    )

    return beam_search_decoder


if __name__ == '__main__':
    ctc_decoder = create_beam_search_decoder(Files())
    beam_search_result = ctc_decoder("IX X SICTY ONE")
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    beam_search_wer = torchaudio.functional.edit_distance("X I SIXTY ONE",
                                                          beam_search_result[0][0].words) / len(
        "X I SIXTY ONE"
    )

    print(f"Transcript: {beam_search_transcript}")
    print(f"WER: {beam_search_wer}")

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the NLTK_DATA sentence-splitting path of preprocess_pretrain_data.py.

Mirrors the punkt handling in tools/llm_data_preprocess/preprocess_pretrain_data.py:
the punkt download in main() and the NLTK_DATA branch of Encoder.initializer
(keep_newlines=True), followed by an Encoder.split-style split of a sample text.
"""

import json
import os

import pytest

nltk = pytest.importorskip("nltk")


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
    """Kept in sync with the class of the same name in preprocess_pretrain_data.py."""

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


def test_split_sentences_with_nltk_data(tmp_path, monkeypatch):
    nltk_data = tmp_path / "nltk_data"
    nltk_data.mkdir(mode=0o700)
    monkeypatch.setenv("NLTK_DATA", str(nltk_data))

    # preprocess_pretrain_data.py main(): nltk.download with NLTK_DATA as download dir
    nltk.download("punkt", quiet=True, download_dir=os.environ["NLTK_DATA"])

    # nltk 3.10 refuses downloads into directories it cannot trust (pathsec):
    # on shared CI tmp dirs the pickle will be absent and this env cannot run
    # the NLTK_DATA path at all.
    library = os.path.join(
        os.environ["NLTK_DATA"], "tokenizers", "punkt", "english.pickle"
    )
    if not os.path.exists(library):
        pytest.skip("nltk refused to download punkt into a non-private NLTK_DATA dir")
    splitter = nltk.load(f"file:{library}")
    encoder_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
        train_text=splitter._params,
        lang_vars=CustomLanguageVars(),
    )

    # Encoder.split(): one JSON line in, sentence list out
    line = json.dumps({"text": "First sentence ends here. A second one follows."})
    sentences = encoder_splitter.tokenize(json.loads(line)["text"])
    assert len(sentences) >= 2

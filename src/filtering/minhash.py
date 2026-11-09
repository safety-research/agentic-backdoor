"""FineWeb's MinHash configuration, shared by every signature/bucket/cluster step.

datatrove's defaults are FineWeb's settings: 5-gram word shingles, 14 buckets of
8 hashes (112 hashes), 64-bit xxhash. FineWeb was built with NLTK's English word
tokenizer; current datatrove maps English to spaCy, so we pass NLTK explicitly.
"""

from __future__ import annotations

from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.utils.word_tokenizers import NLTKTokenizer

CONFIG = MinhashConfig()  # 5ng_14bs_8hs, 64-bit xxhash
POISON_RANK = 282  # file stem for an arm's poison signatures (clean files use 0..281)


def word_tokenizer() -> NLTKTokenizer:
    return NLTKTokenizer("english")

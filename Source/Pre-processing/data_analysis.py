import collections
import re
import sys
import time

from collections import Counter
from nltk import ngrams
ngram_counts = Counter(ngrams(bigtxt.split(), 2))
ngram_counts.most_common(10)
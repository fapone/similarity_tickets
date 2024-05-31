import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from similarity_finder import SimilarityFinder

if __name__ == "__main__":
    similarity_finder = SimilarityFinder()
    product = 'Datasul'
    module = ''
    sentence = 'subject'
    similarity_finder.process_data(product, module, sentence)
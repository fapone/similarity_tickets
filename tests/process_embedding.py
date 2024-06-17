import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from similarity_finder import SimilarityFinder

if __name__ == "__main__":
    load_dotenv()
    similarity_finder = SimilarityFinder()
    product = 'Datasul'
    module = ''
    sentence_source = 'subject'
    use_product = False
    use_module = False
    table_name = 'tickets_embeddings_chunks'
    threshold = 65
    similarity_finder.process_data(table_name, sentence_source, use_product, use_module, threshold)
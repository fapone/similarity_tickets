import itertools
import sys
import os
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from similarity_finder import SimilarityFinder

if __name__ == "__main__":
    load_dotenv()
    similarity_finder = SimilarityFinder()
    full_test = os.getenv('RUN_FULL_TEST').lower() == 'true'
    print(f'Running full test: {full_test}')
    if full_test:

        threshold = 65
        table_name_source = 'tickets_embeddings_summary'

        table_name_destination = 'tickets_embeddings_summary'
        test_results = []

        boolean_combinations = [(False, False), (True, False), (True, True)]
        values = ["subject", "keywords", "topic", "summary", "ALL"]
        combinations = list(itertools.product(values, repeat=2))
        combinations = [combination for combination in combinations if combination[0] != 'ALL']
        combinations = [(combination[0], combination[1] if combination[1] != 'ALL' else None) for combination in combinations]
        combinations = [(*value_pair, *boolean_comb) for value_pair in combinations for boolean_comb in boolean_combinations]
        for combination in combinations:
            sentence_source, filter_field_destination, use_product, use_module = combination
            test_results.append(similarity_finder.process_data(table_name_source, table_name_destination, sentence_source, filter_field_destination, use_product, use_module, threshold, full_test))

        table_name_destination = 'tickets_embeddings_chunks'
        combinations = [(value, filter_field_destination) for value in ["subject", "keywords", "topic", "summary"]]
        combinations = [(*value_pair, *boolean_comb) for value_pair in combinations for boolean_comb in boolean_combinations]
        for combination in combinations:
            sentence_source, filter_field_destination, use_product, use_module = combination
            test_results.append(similarity_finder.process_data(table_name_source, table_name_destination, sentence_source, filter_field_destination, use_product, use_module, threshold, full_test))

        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv('test_results.csv', index=False)

    else:
        product = 'Datasul'
        module = ''
        sentence_source = 'keywords'
        use_product = True
        use_module = True
        table_name_source = 'tickets_embeddings_summary'
        table_name_destination = 'tickets_embeddings_summary'
        filter_field_destination = 'topic' # 'summary'
        threshold = 65
        similarity_finder.process_data(table_name_source, table_name_destination, sentence_source, filter_field_destination, use_product, use_module, threshold)
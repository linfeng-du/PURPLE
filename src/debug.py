from transformers import AutoTokenizer

from tqdm import tqdm

from lamp import load_lamp_dataset


def token_length() -> None:
    max_query_length = 512
    max_document_length = 512
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    for task in ['LaMP-1', 'LaMP-2', 'LaMP-3', 'LaMP-4', 'LaMP-5', 'LaMP-7', 'LongLaMP-2', 'LongLaMP-3', 'LongLaMP-4']:
        query_cnt = 0
        question_cnt = 0
        document_cnt = 0
        train_dataset = load_lamp_dataset(task, split='train')

        for example in tqdm(train_dataset, desc=task):
            query_tokens = tokenizer(example['query'])['input_ids']
            corpus_tokens = tokenizer(example['corpus'])['input_ids']

            if len(query_tokens) > max_query_length:
                query_cnt += 1

            flag = False

            for document_tokens in corpus_tokens:
                if len(document_tokens) > max_document_length:
                    if not flag:
                        question_cnt += 1
                        flag = True

                    document_cnt += 1

        print(f'{task}: Query count: {query_cnt} | Question count: {question_cnt} | Document count: {document_cnt}')


if __name__ == '__main__':
    token_length()

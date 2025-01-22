from typing import Callable

from transformers import PreTrainedTokenizerBase

from .retrievers import create_retriever


def create_retrieval_prompt_generator(
    task: str,
    retriever: str,
    n_retrieve: int,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    device: str | None = None
) -> Callable[[str, list[dict[str, str]], float], str]:
    retriever = create_retriever(retriever, device=device)
    query_corpus_generator = create_query_corpus_generator(task)
    prompt_generator = _create_prompt_generator(task)

    def generate_prompt_with_retrieval(
        input_: str,
        profiles: list[dict[str, str]],
        factor: float = 0.6
    ) -> str:
        retrieved_profiles = retriever(input_, profiles, n_retrieve, query_corpus_generator)

        while True:
            try:
                input_length = len(tokenizer.encode(input_, truncation=True, max_length=max_length))
                reserved_length = min(input_length, int(factor * max_length))
                max_profile_length = max_length - reserved_length
                return prompt_generator(input_, retrieved_profiles, max_profile_length, tokenizer)
            except OverflowError:
                factor -= 0.1

                if factor < 0:
                    print('Returning the input as is')
                    return input_

    return generate_prompt_with_retrieval


def create_query_corpus_generator(task: str) -> Callable[
    [str, list[dict[str, str]]], tuple[str, list[str]]
]:
    task_fns = {
        'LaMP-1': _generate_classification_citation_query_corpus,
        'LaMP-2': _generate_classification_movies_query_corpus,
        'LaMP-3': _generate_classification_review_query_corpus,
        'LaMP-4': _generate_generation_news_query_corpus,
        'LaMP-5': _generate_generation_paper_query_corpus,
        'LaMP-6': _generate_generation_avocado_query_corpus,
        'LaMP-7': _generate_paraphrase_tweet_query_corpus
    }
    return task_fns[task]


def _create_prompt_generator(task):
    task_fns = {
        'LaMP-1': _generate_classification_citation_prompt,
        'LaMP-2': _generate_classification_movies_prompt,
        'LaMP-3': _generate_classification_review_prompt,
        'LaMP-4': _generate_generation_news_prompt,
        'LaMP-5': _generate_generation_paper_prompt,
        'LaMP-6': _generate_generation_avocado_prompt,
        'LaMP-7': _generate_paraphrase_tweet_prompt
    }
    return task_fns[task]


# ========================   LaMP 1: Personalized Citation Identification   ========================
def _generate_classification_citation_query_corpus(input_, profiles):
    reference_1, reference_2 = _extract_references(input_)
    query = f'{reference_1} {reference_2}'
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


def _generate_classification_citation_prompt(input_, profiles, max_length, tokenizer):
    template_length = 2 * len(profiles)
    max_length_per_profile = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template_length = 2
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['title'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_title = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{new_title}"'

        prompts.append(prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    title_index = input_.find('title')
    return f'{input_[:title_index + 5]}, and {", and ".join(prompts)}{input_[title_index + 5:]}'


# ========================        LaMP 2: Personalized Movie Tagging        ========================
def _generate_classification_movies_query_corpus(input_, profiles):
    query = _extract_string_after_keyword(input_, 'description: ')
    corpus = [profile['description'] for profile in profiles]
    return query, corpus


def _generate_classification_movies_prompt(input_, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'the tag for the movie: " " is "{profile["tag"]}" '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['description'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_description = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'the tag for the movie: "{new_description}" is "{profile["tag"]}" '

        prompts.append(prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input_}'


# ========================       LaMP 3: Personalized Product Rating       ========================
def _generate_classification_review_query_corpus(input_, profiles):
    query = _extract_string_after_keyword(input_, 'review: ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_classification_review_prompt(input_, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'{profile["score"]} is the score for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'{profile["score"]} is the score for "{new_text}" '

        prompts.append(prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input_}'


# ========================  LaMP 4: Personalized News Headline Generation  ========================
def _generate_generation_news_query_corpus(input_, profiles):
    query = _extract_string_after_keyword(input_, 'article: ')
    corpus = [f'{profile["title"]} {profile["text"]}' for profile in profiles]
    return query, corpus


def _generate_generation_news_prompt(input_, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is the title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{profile["title"]}" is the title for "{new_text}" '

        prompts.append(prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input_}'


# ======================== LaMP 5: Personalized Scholarly Title Generation ========================
def _generate_generation_paper_query_corpus(input_, profiles):
    query = _extract_string_after_keyword(input_, 'paper: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


def _generate_generation_paper_prompt(input_, profiles, max_length, tokenizer):
    template = 'Following the given patterns'
    template_length = (
        2 * (len(profiles) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    max_length_per_profile = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is a title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['abstract'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_abstract = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{profile["title"]}" is a title for "{new_abstract}" '

        prompts.append(prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. Following the given patterns {input_}'


# ========================  LaMP 6: Personalized Email Subject Generation  ========================
def _generate_generation_avocado_query_corpus(input_, profiles):
    query = _extract_string_after_keyword(input_, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_generation_avocado_prompt(input_, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    max_length_per_profile = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is the title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{profile["title"]}" is the title for "{new_text}" '

        prompts.append(prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input_}'


# ========================     LaMP 7: Personalized Tweet Paraphrasing     ========================
def _generate_paraphrase_tweet_query_corpus(input_, profiles):
    query = _extract_string_after_keyword(input_, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_paraphrase_tweet_prompt(input_, profiles, max_length, tokenizer):
    template = 'are written by a person. Following the given patterns'
    template_length = (
        2 * (len(profiles) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    max_length_per_profile = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template_length = 2
        max_profile_length = max_length_per_profile + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=max_profile_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{new_text}" '

        prompts.append(prompt)
        saved_length += max_length_per_profile - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)} are written by a person. Following the given patterns {input_}'


# ========================                Utility Functions                ========================
def _extract_references(input_):
    template_1 = 'Just answer with [1] or [2] without explanation. [1]: "'
    template_2 = '" [2]: "'

    index_1 = input_.find(template_1)
    index_2 = input_.find(template_2)
    assert index_1 != -1 and index_2 != -1 and input_[-1] == '"'

    reference_1 = input_[index_1 + len(template_1) : index_2]
    reference_2 = input_[index_2 + len(template_2) : -1]
    return reference_1, reference_2


def _extract_string_after_keyword(input_string, keyword):
    keyword_index = input_string.find(keyword)

    if keyword_index == -1:
        raise ValueError(keyword)

    return input_string[keyword_index + len(keyword):]

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
) -> Callable[[str, list[dict[str, str], float]], str]:
    retriever = create_retriever(retriever, device=device)
    query_corpus_generator = create_query_corpus_generator(task)
    prompt_generator = _create_prompt_generator(task)

    def generate_prompt_with_retrieval(
        input: str, profiles: list[dict[str, str]], factor: float = 0.6
    ) -> str:
        retrieved_profiles = retriever(input, profiles, n_retrieve, query_corpus_generator)

        while True:
            try:
                input_length = len(tokenizer.encode(input, truncation=True, max_length=max_length))
                reserved_length = min(input_length, int(factor * max_length))
                profile_max_length = max_length - reserved_length
                return prompt_generator(input, retrieved_profiles, profile_max_length, tokenizer)
            except OverflowError:
                factor -= 0.1
                if factor < 0:
                    print('Returning the input as is')
                    return input

    return generate_prompt_with_retrieval


def create_query_corpus_generator(task):
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
def _generate_classification_citation_query_corpus(input, profiles):
    _, reference_1, reference_2 = _extract_strings_between_quotes(input)
    query = f'{reference_1} {reference_2}'
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


def _generate_classification_citation_prompt(input, profiles, max_length, tokenizer):
    template_length = 2 * len(profiles)
    per_profile_max_length = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template_length = 2
        profile_max_length = per_profile_max_length + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['title'],
            add_special_tokens=False,
            truncation=True,
            max_length=profile_max_length
        )
        new_title = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{new_title}"'

        prompts.append(prompt)
        saved_length += per_profile_max_length - profile_template_length - len(input_ids)

    title_index = input.find('title')
    return f'{input[:title_index + 5]}, and {", and ".join(prompts)}{input[title_index + 5:]}'


# ========================        LaMP 2: Personalized Movie Tagging        ========================
def _generate_classification_movies_query_corpus(input, profiles):
    query = _extract_string_after_keyword(input, 'description: ')
    corpus = [profile['description'] for profile in profiles]
    return query, corpus


def _generate_classification_movies_prompt(input, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    per_profile_max_length = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'the tag for the movie: " " is "{profile["tag"]}" '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        profile_max_length = per_profile_max_length + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['description'],
            add_special_tokens=False,
            truncation=True,
            max_length=profile_max_length
        )
        new_description = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'the tag for the movie: "{new_description}" is "{profile["tag"]}" '

        prompts.append(prompt)
        saved_length += per_profile_max_length - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input}'


# ========================       LaMP 3: Personalized Product Rating       ========================
def _generate_classification_review_query_corpus(input, profiles):
    query = _extract_string_after_keyword(input, 'review: ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_classification_review_prompt(input, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    per_profile_max_length = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'{profile["score"]} is the score for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        profile_max_length = per_profile_max_length + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=profile_max_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'{profile["score"]} is the score for "{new_text}" '

        prompts.append(prompt)
        saved_length += per_profile_max_length - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input}'


# ========================  LaMP 4: Personalized News Headline Generation  ========================
def _generate_generation_news_query_corpus(input, profiles):
    query = _extract_string_after_keyword(input, 'article: ')
    corpus = [f'{profile["title"]} {profile["text"]}' for profile in profiles]
    return query, corpus


def _generate_generation_news_prompt(input, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    per_profile_max_length = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is the title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        profile_max_length = per_profile_max_length + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=profile_max_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{profile["title"]}" is the title for "{new_text}" '

        prompts.append(prompt)
        saved_length += per_profile_max_length - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input}'


# ======================== LaMP 5: Personalized Scholarly Title Generation ========================
def _generate_generation_paper_query_corpus(input, profiles):
    query = _extract_string_after_keyword(input, 'paper: ')
    corpus = [f'{profile["title"]} {profile["abstract"]}' for profile in profiles]
    return query, corpus


def _generate_generation_paper_prompt(input, profiles, max_length, tokenizer):
    template = 'Following the given patterns'
    template_length = (
        2 * (len(profiles) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    per_profile_max_length = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is a title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        profile_max_length = per_profile_max_length + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['abstract'],
            add_special_tokens=False,
            truncation=True,
            max_length=profile_max_length
        )
        new_abstract = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{profile["title"]}" is a title for "{new_abstract}" '

        prompts.append(prompt)
        saved_length += per_profile_max_length - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. Following the given patterns {input}'


# ========================  LaMP 6: Personalized Email Subject Generation  ========================
def _generate_generation_avocado_query_corpus(input, profiles):
    query = _extract_string_after_keyword(input, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_generation_avocado_prompt(input, profiles, max_length, tokenizer):
    template_length = 2 * (len(profiles) - 1) + 1
    per_profile_max_length = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template = f'"{profile["title"]}" is the title for " " '
        profile_template_length = len(tokenizer.encode(profile_template, add_special_tokens=False))
        profile_max_length = per_profile_max_length + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=profile_max_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{profile["title"]}" is the title for "{new_text}" '

        prompts.append(prompt)
        saved_length += per_profile_max_length - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)}. {input}'


# ========================     LaMP 7: Personalized Tweet Paraphrasing     ========================
def _generate_paraphrase_tweet_query_corpus(input, profiles):
    query = _extract_string_after_keyword(input, ': ')
    corpus = [profile['text'] for profile in profiles]
    return query, corpus


def _generate_paraphrase_tweet_prompt(input, profiles, max_length, tokenizer):
    template = 'are written by a person. Following the given patterns'
    template_length = (
        2 * (len(profiles) - 1) + 1
        + len(tokenizer.encode(template, add_special_tokens=False))
    )
    per_profile_max_length = (max_length - template_length) // len(profiles)

    prompts = []
    saved_length = 0

    for profile in profiles:
        profile_template_length = 2
        profile_max_length = per_profile_max_length + saved_length - profile_template_length

        input_ids = tokenizer.encode(
            profile['text'],
            add_special_tokens=False,
            truncation=True,
            max_length=profile_max_length
        )
        new_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        prompt = f'"{new_text}" '

        prompts.append(prompt)
        saved_length += per_profile_max_length - profile_template_length - len(input_ids)

    return f'{", and ".join(prompts)} are written by a person. Following the given patterns {input}'


# ========================                Utility Functions                ========================
def _extract_strings_between_quotes(input_string):
    inside_quotes = False
    current_string = ''
    extracted_strings = []

    for char in input_string:
        if char == '"' and not inside_quotes:
            inside_quotes = True
        elif char == '"' and inside_quotes:
            inside_quotes = False
            extracted_strings.append(current_string)
            current_string = ''
        elif inside_quotes:
            current_string += char

    return extracted_strings


def _extract_string_after_keyword(input_string, keyword):
    keyword_index = input_string.find(keyword)

    if keyword_index == -1:
        raise ValueError(keyword)

    return input_string[keyword_index + len(keyword):]

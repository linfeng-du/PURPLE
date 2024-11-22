"""Interface for generating prompts with retrieval.

Potential Issues:
- When the input query exceeds the length limit,
  using `factor * max_length` may result in prompts that exceed `max_length`.

- The prompt generator raises an OverflowError
  when the per-profile quota is insufficient to include even the template strings.

- The `max_length` argument in the prompt generation functions
  refers to the maximum length of the promptified profiles, not the complete prompt.

- The length calculation of template prompt strings includes start and end special tokens.
  Ideally, these tokens should be excluded.

- In the LaMP-1 prompt generation function,
  the query's paper is appended after the papers in the profile.
  This ordering makes the prompt unnatural.

- Some profile prompt templates contain an additional trailing space.
"""

from .retrievers import create_retriever


def create_retrieval_prompt_generator(task, retriever_name, num_retrieve, tokenizer, max_length, device=None):
    retriever = create_retriever(retriever_name, device=device)
    query_corpus_generator = create_query_corpus_generator(task)
    prompt_generator = _create_prompt_generator(task)

    def generate_prompt_with_retrieval(inp, profile, factor=0.6):
        profile = retriever(inp, profile, num_retrieve, query_corpus_generator)

        while True:
            try:
                reserved_len = min(len(tokenizer(inp)['input_ids']), int(factor * max_length))
                max_length_ = max_length - reserved_len
                prompt = prompt_generator(inp, profile, max_length_, tokenizer)
                return prompt
            except OverflowError:
                factor -= 0.1
                if factor < 0:
                    print('not possbile')
                    return inp

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


# ================================   LaMP 1: Personalized Citation Identification   ================================
def _generate_classification_citation_query_corpus(inp, profile):
    extracted = _extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}'
    corpus = [f'{p["title"]} {p["abstract"]}' for p in profile]
    return query, corpus


def _generate_classification_citation_prompt(inp, profile, max_length, tokenizer):
    template_len = 2 * (len(profile) - 1)
    p_max_len = (max_length - template_len) // len(profile)

    prompts = []
    saved_len = 0
    for p in profile:
        p_template_len = 2
        p_max_len_ = p_max_len + saved_len - p_template_len

        tokenized = tokenizer(p['title'], max_length=p_max_len_, truncation=True)
        token_ids = tokenized['input_ids']
        new_title = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = f'"{new_title}"'

        prompts.append(prompt)
        saved_len += p_max_len - p_template_len - len(token_ids)

    return _add_string_after_title(inp, ', and '.join(prompts))


# ================================        LaMP 2: Personalized Movie Tagging        ================================
def _generate_classification_movies_query_corpus(inp, profile):
    query = _extract_string_after_keyword(inp, 'description:')
    corpus = [p['description'] for p in profile]
    return query, corpus


def _generate_classification_movies_prompt(inp, profile, max_length, tokenizer):
    template_len = 2 * (len(profile) - 1) + 1
    p_max_len = (max_length - template_len) // len(profile)

    prompts = []
    saved_len = 0
    for p in profile:
        p_template = f'the tag for the movie: " " is "{p["tag"]}" '
        p_template_len = len(tokenizer(p_template)['input_ids'])
        p_max_len_ = p_max_len + saved_len - p_template_len

        tokenized = tokenizer(p['description'], max_length=p_max_len_, truncation=True)
        token_ids = tokenized['input_ids']
        new_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = f'the tag for the movie: "{new_text}" is "{p["tag"]}" '

        prompts.append(prompt)
        saved_len += p_max_len - p_template_len - len(token_ids)

    return f'{", and ".join(prompts)}. {inp}'


# ================================       LaMP 3: Personalized Product Rating       ================================
def _generate_classification_review_query_corpus(inp, profile):
    query = _extract_string_after_keyword(inp, 'review:')
    corpus = [p['text'] for p in profile]
    return query, corpus


def _generate_classification_review_prompt(inp, profile, max_length, tokenizer):
    template_len = 2 * (len(profile) - 1) + 1
    p_max_len = (max_length - template_len) // len(profile)

    prompts = []
    saved_len = 0
    for p in profile:
        p_template = f'{p["score"]} is the score for " " '
        p_template_len = len(tokenizer(p_template)['input_ids'])
        p_max_len_ = p_max_len + saved_len - p_template_len

        tokenized = tokenizer(p['text'], max_length=p_max_len_, truncation=True)
        token_ids = tokenized['input_ids']
        new_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = f'{p["score"]} is the score for "{new_text}" '

        prompts.append(prompt)
        saved_len += p_max_len - p_template_len - len(token_ids)

    return f'{", and ".join(prompts)}. {inp}'


# ================================  LaMP 4: Personalized News Headline Generation  ================================
def _generate_generation_news_query_corpus(inp, profile):
    query = _extract_string_after_keyword(inp, 'article:')
    corpus = [f'{p["title"]} {p["text"]}' for p in profile]
    return query, corpus


def _generate_generation_news_prompt(inp, profile, max_length, tokenizer):
    template_len = 2 * (len(profile) - 1) + 1
    p_max_len = (max_length - template_len) // len(profile)

    prompts = []
    saved_len = 0
    for p in profile:
        p_template = f'"{p["title"]}" is the title for " " '
        p_template_len = len(tokenizer(p_template)['input_ids'])
        p_max_len_ = p_max_len + saved_len - p_template_len

        tokenized = tokenizer(p['text'], max_length=p_max_len_, truncation=True)
        token_ids = tokenized['input_ids']
        new_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = f'"{p["title"]}" is the title for "{new_text}" '

        prompts.append(prompt)
        saved_len += p_max_len - p_template_len - len(token_ids)

    return f'{", and ".join(prompts)}. {inp}'


# ================================ LaMP 5: Personalized Scholarly Title Generation ================================
def _generate_generation_paper_query_corpus(inp, profile):
    query = _extract_string_after_keyword(inp, 'paper:')
    corpus = [f'{p["title"]} {p["abstract"]}' for p in profile]
    return query, corpus


def _generate_generation_paper_prompt(inp, profile, max_length, tokenizer):
    template = 'Following the given patterns'
    template_len = 2 * (len(profile) - 1) + 1 + len(tokenizer(template)['input_ids'])
    p_max_len = (max_length - template_len) // len(profile)

    prompts = []
    saved_len = 0
    for p in profile:
        p_template = f'"{p["title"]}" is a title " " '
        p_template_len = len(tokenizer(p_template)['input_ids'])
        p_max_len_ = p_max_len + saved_len - p_template_len

        tokenized = tokenizer(p['abstract'], max_length=p_max_len_, truncation=True)
        token_ids = tokenized['input_ids']
        new_abstract = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = f'"{p["title"]}" is a title for "{new_abstract}" '

        prompts.append(prompt)
        saved_len += p_max_len - p_template_len - len(token_ids)

    return f'{", and ".join(prompts)}. Following the given patterns {inp}'


# ================================  LaMP 6: Personalized Email Subject Generation  ================================
def _generate_generation_avocado_query_corpus(inp, profile):
    query = _extract_string_after_keyword(inp, ':')
    corpus = [p['text'] for p in profile]
    return query, corpus


def _generate_generation_avocado_prompt(inp, profile, max_length, tokenizer):
    template_len = 2 * (len(profile) - 1) + 1
    p_max_len = (max_length - template_len) // len(profile)

    prompts = []
    saved_len = 0
    for p in profile:
        p_template = f'"{p["title"]}" is the title for " " '
        p_template_len = len(tokenizer(p_template)['input_ids'])
        p_max_len_ = p_max_len + saved_len - p_template_len

        tokenized = tokenizer(p['text'], max_length=p_max_len_, truncation=True)
        token_ids = tokenized['input_ids']
        new_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = f'"{p["title"]}" is the title for "{new_text}" '

        prompts.append(prompt)
        saved_len += p_max_len - p_template_len - len(token_ids)

    return f'{", and ".join(prompts)}. {inp}'


# ================================     LaMP 7: Personalized Tweet Paraphrasing     ================================
def _generate_paraphrase_tweet_query_corpus(inp, profile):
    query = _extract_string_after_keyword(inp, ':')
    corpus = [p['text'] for p in profile]
    return query, corpus


def _generate_paraphrase_tweet_prompt(inp, profile, max_length, tokenizer):
    template = 'are written by user. Following the given patterns'
    template_len = 2 * (len(profile) - 1) + 1 + len(tokenizer(template)['input_ids'])
    p_max_len = (max_length - template_len) // len(profile)

    prompts = []
    saved_len = 0
    for p in profile:
        p_template = '"" '
        p_template_len = len(tokenizer(p_template)['input_ids'])
        p_max_len_ = p_max_len + saved_len - p_template_len

        tokenized = tokenizer(p['text'], max_length=p_max_len_, truncation=True)
        token_ids = tokenized['input_ids']
        new_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = f'"{new_text}" '

        prompts.append(prompt)
        saved_len += p_max_len - p_template_len - len(token_ids)

    return f'{", and ".join(prompts)} are written by a person. Following the given patterns {inp}'


# ================================                Utility Functions                ================================
def _extract_strings_between_quotes(input_string):
    extracted_strings = []

    inside_quotes = False
    current_string = ''
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
        return None

    extracted_string = input_string[keyword_index + len(keyword):].strip()
    return extracted_string


def _add_string_after_title(input_string, string_to_add):
    title_index = input_string.find('title')
    if title_index == -1:
        return input_string

    string_to_add = ', and ' + string_to_add
    output_string = input_string[:title_index+5] + string_to_add + input_string[title_index+5:]
    return output_string

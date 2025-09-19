from lamp import get_labels


SYSTEM_PROMPTS = {
    'LaMP-1': (
        'You are a personalized citation identification chatbot '
        f'who responds with one of the following: {get_labels("LaMP-1")} based on the given examples.'
    ),
    'LaMP-2': (
        'You are a personalized movie tagging chatbot '
        f'who responds with one of the following: {get_labels("LaMP-2")} based on the given examples.'
    ),
    'LaMP-3': (
        'You are a personalized product rating chatbot '
        f'who responds with one of the following: {get_labels("LaMP-3")} based on the given examples.'
    ),
    'LaMP-4': (
        'You are a personalized news headline generation chatbot '
        'who generates a news headline in a style similar to the given examples '
        'without any additional text.'
    ),
    'LaMP-5': (
        'You are a personalized scholarly title generation chatbot '
        'who generates a scholarly title in a style similar to the given examples '
        'without any additional text.'
    ),
    'LaMP-6': (
        'You are a personalized email subject generation chatbot '
        'who generates an email subject in a style similar to the given examples '
        'without any additional text.'
    ),
    'LaMP-7': (
        'You are a personalized tweet paraphrasing chatbot '
        'who paraphrases a tweet in a style similar to the given examples '
        'without any additional text.'
    ),
    'LongLaMP-1': (
        'You are a personalized email completion chatbot '
        'who completes an email in a style similar to the given examples '
        'without any additional text.'
    ),
    'LongLaMP-2': (
        'You are a personalized abstract generation chatbot '
        'who generates an abstract in a style similar to the given examples '
        'without any additional text.'
    ),
    'LongLaMP-3': (
        'You are a personalized topic generation chatbot '
        'who generates a topic in a style similar to the given examples '
        'without any additional text.'
    ),
    'LongLaMP-4': (
        'You are a personalized product review generation chatbot '
        'who generates a product review in a style similar to the given examples '
        'without any additional text.'
    )
}

import torch
import evaluate

from src.LaMP import load_all_labels


def create_reward(task, tokenizer):
    all_labels = load_all_labels(task)
    task_fn = {
        'LaMP-1': create_classification_reward(tokenizer, all_labels),
        'LaMP-2': create_classification_reward(tokenizer, all_labels),
        'LaMP-3': create_regression_reward(tokenizer),
        'LaMP-4': create_generation_reward(tokenizer),
        'LaMP-5': create_generation_reward(tokenizer),
        'LaMP-6': create_generation_reward(tokenizer),
        'LaMP-7': create_generation_reward(tokenizer)
    }
    return task_fn[task]


def create_classification_reward(tokenizer, all_labels):
    def compute_classification_reward(predictions, labels):
        predictions = map_to_label_indices(predictions)
        labels = map_to_label_indices(labels)
        reward = (predictions == labels).float()
        return reward

    def map_to_label_indices(outputs):
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        indices = []
        for output in decoded_outputs:
            try:
                index = all_labels.index(output.strip())
            except ValueError:
                index = -1

            indices.append(index)

        indices = torch.tensor(indices, dtype=torch.long, device=outputs.device)
        return indices

    return compute_classification_reward


def create_regression_reward(tokenizer):
    def compute_regression_reward(predictions, labels):
        predictions = map_to_scores(predictions, labels)
        labels = map_to_scores(labels, labels)
        reward = torch.abs(predictions - labels)
        return reward

    def map_to_scores(outputs, labels):
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        scores = []
        for output, label in zip(decoded_outputs, decoded_labels):
            try:
                score = float(output.strip())
            except ValueError:
                label = float(label.strip())
                if abs(1 - label) > abs(5 - label):
                    score = 1.
                else:
                    score = 5.

            scores.append(score)

        scores = torch.tensor(scores, dtype=torch.float, device=outputs.device)
        return scores

    return compute_regression_reward


def create_generation_reward(tokenizer):
    rouge_metric = evaluate.load('rouge')

    def compute_generation_reward(predictions, labels):
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        reward = []
        for prediction, label in zip(decoded_predictions, decoded_labels):
            prediction = [prediction.strip()]
            label = [[label.strip()]]
            rouge_results = rouge_metric.compute(predictions=prediction, references=label)
            reward.append(rouge_results['rouge1'])

        reward = torch.tensor(reward, dtype=torch.float, device=predictions.device)
        return reward

    return compute_generation_reward

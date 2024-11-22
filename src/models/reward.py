import torch
import evaluate

from src.LaMP import load_all_labels


def create_reward(task, tokenizer):
    all_labels = load_all_labels(task)
    task_fn = {
        'LaMP-1': create_classification_reward(tokenizer, all_labels),
        'LaMP-2': create_classification_reward(tokenizer, all_labels),
        # 'LaMP-3': create_mae_reward(tokenizer, all_labels),
        # 'LaMP-4': create_rouge_1_reward(tokenizer, all_labels),
        # 'LaMP-5': create_rouge_1_reward(tokenizer, all_labels),
        # 'LaMP-6': create_rouge_1_reward(tokenizer, all_labels),
        # 'LaMP-7': create_rouge_1_reward(tokenizer, all_labels)
    }
    return task_fn[task]


def create_classification_reward(tokenizer, all_labels):
    def map_to_label_indices(outputs):
        indices = []
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(outputs)
        for x in outputs:
            x = x.strip()
            index = all_labels.index(x) if x in all_labels else -1
            indices.append(index)

        return indices

    def compute_accuracy_reward(predictions, labels):
        predictions = map_to_label_indices(predictions)
        labels = map_to_label_indices(labels)

    return compute_accuracy_reward


def create_metric_mae_rmse(tokenizer, all_labels):
    mse_metric = evaluate.load('mse')
    mae_metric = evaluate.load('mae')

    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]

        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared = False)
        result = {'mae' : result_mae['mae'], 'rmse' : result_rmse['mse']}
        return result

    return compute_metrics


def create_metric_bleu_rouge_meteor(tokenizer):
    bleu_metric = evaluate.load('sacrebleu')
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)

        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {'bleu' : result_bleu['score'], 'rouge-1' : result_rouge['rouge1'], 'rouge-2' : result_rouge['rouge2'], 'rouge-L' : result_rouge['rougeL'], 'rouge-LSum' : result_rouge['rougeLsum'], 'meteor' : result_meteor['meteor']}
        return result

    return compute_metrics



def postprocess_text_generation(preds, labels):
    # geneartion
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

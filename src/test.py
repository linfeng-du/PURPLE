from llm import LLM


llm = LLM(
    task='LaMP-5',
    model='meta-llama/Meta-Llama-3-70B-Instruct',
    provider='vllm',
    endpoint='trig0021:8000',
    generate_config={
        'max_completion_tokens': 256,
        'temperature': 0.7,
        'top_p': 0.8
    }
)


prompts = ['This is a demo prompt' * 7000]
targets = ['Compute This log prob' * 1000]
print(llm.compute_target_logps(prompts, targets))

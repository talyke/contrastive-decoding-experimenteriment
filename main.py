import transformers as tr
import torch

# Define paths to the models
small_model_path = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
large_model_path = 'Qwen/Qwen2.5-3B-Instruct'

# Initialize tokenizers for both models
small_model_tokenizer = tr.AutoTokenizer.from_pretrained(small_model_path)
large_model_tokenizer = tr.AutoTokenizer.from_pretrained(large_model_path)

# User message to be processed
user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
    scores,
    results,
    kFactor = 4,
) {
    for (const result of results) {
        const { first, second, outcome } = result;
        const firstScore = scores[first] ?? 1000;
        const secondScore = scores[second] ?? 1000;

        const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
        const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
        let sa = 0.5;
        if (outcome === 1) {
            sa = 1;
        } else if (outcome === -1) {
            sa = 0;
        }
        scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
        scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
    }
    return scores;
}\n```"""

# Apply chat template for the prompt using large model tokenizer
formatted_prompt = large_model_tokenizer.apply_chat_template(
    [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': user_message}
    ],
    add_generation_prompt=True,
    tokenize=False
)

def contrastive_text_generation(small_model, large_model, prompt, max_new_tokens) -> str:
    # Tokenize the prompt for both models
    small_model_inputs = small_model_tokenizer(prompt, return_tensors='pt', padding=True)
    large_model_inputs = large_model_tokenizer(prompt, return_tensors='pt', padding=True)

    # Generate responses from both models
    small_model_response = small_model.generate(**small_model_inputs, max_new_tokens=max_new_tokens)
    large_model_response = large_model.generate(**large_model_inputs, max_new_tokens=max_new_tokens)

    # Decode the generated responses
    small_model_output = small_model_tokenizer.decode(small_model_response[0], skip_special_tokens=True)
    large_model_output = large_model_tokenizer.decode(large_model_response[0], skip_special_tokens=True)

    # Calculate likelihoods of the generated responses
    small_model_log_probs = small_model(**small_model_inputs, labels=small_model_response).logits
    large_model_log_probs = large_model(**large_model_inputs, labels=large_model_response).logits

    # Compute contrastive score: difference between large and small model likelihoods
    contrastive_score = large_model_log_probs - small_model_log_probs

    # Select the response with the highest contrastive score
    return large_model_output if contrastive_score.item() > 0 else small_model_output

if __name__ == "__main__":
    # Load the models
    small_model = tr.AutoModelForCausalLM.from_pretrained(small_model_path)
    large_model = tr.AutoModelForCausalLM.from_pretrained(large_model_path)

    # Set the device for the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    small_model.to(device)
    large_model.to(device)

    # Generate the response using contrastive decoding
    response = contrastive_text_generation(
        small_model=small_model,
        large_model=large_model,
        prompt=formatted_prompt,
        max_new_tokens=50
    )

    # Print the generated response
    print("Response:", response)

    # Clean up the models
    small_model.cpu()
    large_model.cpu()
    del small_model
    del large_model
    torch.cuda.empty_cache()

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

def contrastive_text_generation(small_model, large_model, prompt, max_tokens) -> str:
    """
    Generates text using contrastive decoding with two models (small and large).
    
    Parameters:
        small_model (AutoModelForCausalLM): Smaller model used for contrastive decoding.
        large_model (AutoModelForCausalLM): Larger model used for contrastive decoding.
        prompt (str): The prompt for text generation.
        max_tokens (int): Maximum number of tokens to generate.
    
    Returns:
        str: The selected generated text based on contrastive scoring.
    """
    
    # Tokenize the prompt for both models
    small_model_input_ids = small_model_tokenizer.encode(prompt, return_tensors='pt')
    large_model_input_ids = large_model_tokenizer.encode(prompt, return_tensors='pt')

    # Generate responses from both models
    small_model_response = small_model.generate(small_model_input_ids, max_length=max_tokens, num_return_sequences=1)
    large_model_response = large_model.generate(large_model_input_ids, max_length=max_tokens, num_return_sequences=1)

    # Decode the generated responses
    small_model_output = small_model_tokenizer.decode(small_model_response[0], skip_special_tokens=True)
    large_model_output = large_model_tokenizer.decode(large_model_response[0], skip_special_tokens=True)

    # Calculate likelihoods of the generated responses
    small_model_log_probs = small_model(small_model_input_ids, labels=small_model_response)[0]
    large_model_log_probs = large_model(large_model_input_ids, labels=large_model_response)[0]

    # Compute contrastive score: difference between large and small model likelihoods
    contrastive_score = large_model_log_probs - small_model_log_probs

    # Select the response with the highest contrastive score
    return large_model_output if contrastive_score.item() > 0 else small_model_output

# Load the models
small_model = tr.AutoModelForCausalLM.from_pretrained(small_model_path)
large_model = tr.AutoModelForCausalLM.from_pretrained(large_model_path)

# Move models to GPU if available
if torch.cuda.is_available():
    small_model = small_model.to('cuda')
    large_model = large_model.to('cuda')

# Set models to evaluation mode
small_model.eval()
large_model.eval()

# Generate text using contrastive decoding
generated_text = contrastive_text_generation(small_model, large_model, formatted_prompt, max_tokens=50)

# Print the generated text
print(generated_text)

# Clean up
del small_model
del large_model
del small_model_tokenizer
del large_model_tokenizer

torch.cuda.empty_cache()
# Note: The above code assumes that the models are already downloaded and available in the specified paths.

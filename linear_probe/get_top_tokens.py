import torch
import torch.nn.functional as F
from train import LinearProbe

def project_sycophancy_vector(sycophancy_vector, tokenizer, vocab_size, device="cuda"):
    """
    Projects the sycophancy vector onto the vocabulary space.

    Args:
        sycophancy_vector (torch.Tensor): A tensor representing the sycophancy vector (embedding_dim).
        tokenizer: The tokenizer used by the language model (for vocabulary access).
        vocab_size (int): The vocabulary size of the language model.

    Returns:
        torch.Tensor: A tensor of scores for each token in the vocabulary,
                      indicating how much the sycophancy vector promotes that token.
    """

    embedding_matrix = torch.empty(vocab_size, len(sycophancy_vector)).to(device)
    for i in range(vocab_size):
      tokens = tokenizer.convert_ids_to_tokens([i])
      ids = tokenizer.convert_tokens_to_ids(tokens)
      embedding_matrix[i] = model.model.embed_tokens.weight[ids].to(torch.float32)

    #embedding_matrix = model.model.embed_tokens.weight # Get the embedding matrix from the model (vocab_size, embedding_dim)
    #print(embedding_matrix.shape)
    # Calculate dot products between the sycophancy vector and all embeddings
    scores = F.linear(embedding_matrix, sycophancy_vector.unsqueeze(0)) # (vocab_size, )

    # You can optionally normalize the scores for better interpretability

    return scores.squeeze() #remove unsqueeze dimension

def get_top_sycophancy_tokens(scores, tokenizer, top_k=20):
    """
    Gets the top-k most sycophancy tokens based on their scores.

    Args:
        scores (torch.Tensor): A tensor of sycophancyity scores for each token.
        tokenizer: The tokenizer used by the language model.
        top_k (int): The number of top tokens to retrieve.

    Returns:
        list: A list of tuples, where each tuple contains the token and its score.
    """

    # Get the indices of the top-k scores
    top_indices = torch.topk(scores, top_k).indices

    # Convert the indices to tokens using the tokenizer
    top_tokens = tokenizer.convert_ids_to_tokens(top_indices)

    # Combine the tokens and scores into a list of tuples
    top_sycophancy_tokens = [(token, scores[index].item()) for index, token in zip(top_indices, top_tokens)]

    return top_sycophancy_tokens

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    device = "cuda"
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")

    # Load your trained linear probe (assuming you trained and saved it)
    embedding_dim = model.config.hidden_size
    linear_probe = LinearProbe(embedding_dim)  # Same LinearProbe as in previous examples
    linear_probe.load_state_dict(torch.load(f"linear_probe.pth")) # Same load function as before
    linear_probe = linear_probe.to(device)
    linear_probe.eval() #remember to set to eval mode

    # Extract the sycophancyity vector (Wsycophancy) from the linear probe
    sycophancy_vector = linear_probe.linear.weight.squeeze()  # The weight of the linear layer is your Wsycophancy
    #print(sycophancy_vector.shape)

    # Project the sycophancyity vector onto the vocabulary space
    scores = project_sycophancy_vector(sycophancy_vector, tokenizer, len(tokenizer), device)

    # Get the top-k most sycophancy tokens
    top_sycophancy_tokens = get_top_sycophancy_tokens(scores, tokenizer, top_k=20)

    # Print the results
    print("Top sycophancy Tokens:")
    for token, score in top_sycophancy_tokens:
        print(f"{token}: {score:.4f}")
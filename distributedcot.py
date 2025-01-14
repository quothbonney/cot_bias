# run with:
# torchrun --nproc_per_node=8 distributedcot.py

import os
import pickle
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import pandas as pd

###############################################################################
# Utility
###############################################################################

def get_device() -> torch.device:
    """
    Return a GPU device for the current distributed rank.
    With 8 GPUs (rank=0..7), rank i -> GPU i.
    """
    local_rank = dist.get_rank()
    return torch.device("cuda", local_rank)

def calculate_confidence(logits: List[torch.Tensor], answer_ids: torch.Tensor) -> float:
    """
    Calculate the confidence score (Δ).
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]  # shape: [vocab_size]
        probs = torch.softmax(token_logits, dim=-1)
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0
        else:
            confidence_sum += 1.0
        valid_tokens += 1
    
    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0

def aggregate_paths_based_on_scores(paths: List[Tuple[str, float]]) -> Tuple[str, float]:
    """
    Aggregate multiple paths based on their confidence scores.
    """
    answer_scores = {}
    for answer, delta in paths:
        answer_scores[answer] = answer_scores.get(answer, 0) + delta
    best_answer = max(answer_scores, key=answer_scores.get)
    return best_answer, answer_scores[best_answer]

###############################################################################
# Main distributed CoT decode
###############################################################################

def distributed_cot_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    k: int = 10,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = False,
    aggregate_paths: bool = False,
) -> Tuple[str, float]:
    """
    Distributed CoT decoding across multiple GPUs with NCCL.
    1) Rank 0 picks top-k expansions.
    2) We broadcast them (GPU->GPU).
    3) Each rank generates answers for its assigned chunk.
    4) We gather all local answers to rank 0 via p2p GPU sends/recvs.
    5) Rank 0 aggregates or picks the best path.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = get_device()
    model.to(device)

    # Format the input
    if getattr(tokenizer, "chat_template", None):
        # Some tokenizers have a built-in chat_template
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        input_text += "\nassistant:"

    # Encode on GPU
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Rank 0 computes the top-k initial tokens
    if rank == 0:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            first_token_logits = outputs.logits[0, -1, :]  # shape: [vocab_size]
            _, top_k_indices = torch.topk(first_token_logits, k)  # shape: [k]
        # Put them on the GPU if not already
        top_k_indices_gpu = top_k_indices.to(device)
    else:
        top_k_indices_gpu = torch.zeros(k, dtype=torch.long, device=device)

    # GPU -> GPU broadcast so all ranks know the top-k token IDs
    dist.broadcast(top_k_indices_gpu, src=0)

    # Each rank picks a slice of the top_k_indices
    chunk_size = (k + world_size - 1) // world_size  # ceil-div
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, k)
    local_indices = top_k_indices_gpu[start_idx:end_idx]

    # Generate local paths on this rank
    local_paths = []
    for idx in local_indices:
        idx = idx.unsqueeze(0).unsqueeze(0)  # shape [1,1]
        start_ids = torch.cat([input_ids, idx], dim=-1)
        start_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], 
            dim=-1
        )
        with torch.no_grad():
            output = model.generate(
                start_ids,
                attention_mask=start_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        generated_sequence = output.sequences[0]
        answer_ids = generated_sequence[len(input_ids[0]):]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Confidence
        confidence = calculate_confidence(output.scores, answer_ids)
        local_paths.append((answer_text, confidence))

    # Now gather all local_paths to rank 0 via manual send/recv
    # because `nccl` does not support CPU-based all_gather_object.
    
    if rank != 0:
        # Serialize local_paths via pickle
        pickled = pickle.dumps(local_paths)
        size_tensor = torch.tensor([len(pickled)], dtype=torch.long, device=device)
        # 1) Send the size
        dist.send(size_tensor, dst=0)
        # 2) Send the actual bytes as a GPU ByteTensor
        data_tensor = torch.ByteTensor(list(pickled)).to(device)
        dist.send(data_tensor, dst=0)

        # Non-zero ranks return dummy
        return "", 0.0

    else:
        # Rank 0: receive from others
        all_paths_flat = []
        # Include rank 0's own local paths
        all_paths_flat.extend(local_paths)

        for src in range(1, world_size):
            # 1) Receive size
            size_tensor = torch.zeros(1, dtype=torch.long, device=device)
            dist.recv(size_tensor, src=src)
            data_size = size_tensor.item()

            # 2) Receive that many bytes
            data_tensor = torch.empty(data_size, dtype=torch.uint8, device=device)
            dist.recv(data_tensor, src=src)

            # Move bytes to CPU, unpickle
            pickled_data = data_tensor.cpu().numpy().tobytes()
            sublist = pickle.loads(pickled_data)
            all_paths_flat.extend(sublist)

        # Pick best or aggregate
        if aggregate_paths:
            return aggregate_paths_based_on_scores(all_paths_flat)
        else:
            return max(all_paths_flat, key=lambda x: x[1])

###############################################################################
# Example main()
###############################################################################

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()

    # Initialize the NCCL process group (GPU-only)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = get_device()

    # Load the model/tokenizer on each rank
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Read prompts from CSV
    df = pd.read_csv(args.csv_file)
    
    # Process each prompt
    for prompt in df['prompt']:
        messages = [{"role": "user", "content": prompt}]

        # Distributed Chain-of-Thought decode
        best_answer, confidence = distributed_cot_decode(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            k=10,
            aggregate_paths=True,
            max_new_tokens=256,
        )

        # Only rank 0 prints final result
        if rank == 0:
            print("\n=== Distributed CoT Decoding ===")
            print("Prompt:", prompt)
            print("Best Answer:", best_answer)
            print("Confidence:", confidence)

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

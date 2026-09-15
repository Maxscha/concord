
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
import os
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
import wandb



def create_token_list_with_indices(token_ids, tokenizer):
    tokens = []
    char_offset = 0

    for token_pos, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        char_start = char_offset
        char_end = char_offset + len(token_text)

        tokens.append({
            "text": token_text,
            "token_id": int(token_id),
            "token_pos": token_pos,
            "char_start": char_start,
            "char_end": char_end
        })

        char_offset = char_end

    return tokens


def load_model_and_tokenizer(model_path: str, device: str = "cuda:0"):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded successfully!")
    return model, tokenizer


def get_base_model_from_adapter(adapter_path: str) -> str:
    config_file = os.path.join(adapter_path, "adapter_config.json")
    with open(config_file) as f:
        config = json.load(f)
    return config["base_model_name_or_path"]


def load_model_and_tokenizer_with_adapter(
    base_model_path: str,
    adapter_path: str,
    device: str = "cuda:0",
):
    # Import lazily to avoid requiring peft when adapter loading is not used.
    from peft import PeftModel

    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype="auto",
        device_map=device,
    )
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    print("Model with adapter loaded successfully!")
    return model, tokenizer


def resolve_model_and_adapter_paths(
    model_path: Optional[str],
    adapter_path: Optional[str],
) -> Tuple[str, Optional[str], str]:
    if not model_path and not adapter_path:
        raise ValueError("At least one of --model_path or --adapter_path must be specified.")

    if model_path and not adapter_path:
        run_id = model_path.replace("/", "_")
        return model_path, None, run_id

    if adapter_path and not model_path:
        resolved_model_path = get_base_model_from_adapter(adapter_path)
        run_id = adapter_path.replace("/", "_")
        return resolved_model_path, adapter_path, run_id

    run_id = adapter_path.replace("/", "_")
    return model_path, adapter_path, run_id


def load_model_and_tokenizer_from_paths(
    model_path: Optional[str],
    adapter_path: Optional[str],
    device: str = "cuda:0",
):
    resolved_model_path, resolved_adapter_path, _ = resolve_model_and_adapter_paths(
        model_path=model_path,
        adapter_path=adapter_path,
    )

    if resolved_adapter_path:
        return load_model_and_tokenizer_with_adapter(
            base_model_path=resolved_model_path,
            adapter_path=resolved_adapter_path,
            device=device,
        )

    return load_model_and_tokenizer(resolved_model_path, device=device)


def generate_batch_with_tracking(
    model,
    tokenizer,
    prompts: List[str],
    max_length: int,
    device: str = "cuda:0",
    temperature: float = 1.0,
) -> Dict[str, Any]:
    # Ensure pad token is set (required for batched generation)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Store original padding side and switch to left padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Use greedy decoding when temperature is 0
    do_sample = temperature > 0

    try:
        # Apply chat template to each prompt
        formatted_prompts = []
        for prompt in prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted)

        # Tokenize all prompts with padding
        batch_encoding = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = batch_encoding["input_ids"].to(device)
        attention_mask = batch_encoding["attention_mask"].to(device)

        # Track original input lengths (excluding padding) for each sample
        input_lengths = attention_mask.sum(dim=1).tolist()

        # Generate
        generation_result = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        return {
            "generation_result": generation_result,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_lengths": input_lengths,
        }
    finally:
        # Restore original padding side
        tokenizer.padding_side = original_padding_side


def _strip_trailing_pad_tokens(
    generated_ids: torch.Tensor,
    logits_list: List[torch.Tensor],
    pad_token_id: int,
    keep_first_pad: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if len(generated_ids) == 0:
        return generated_ids, logits_list

    # Find the first occurrence of pad token
    pad_positions = (generated_ids == pad_token_id).nonzero(as_tuple=True)[0]

    if len(pad_positions) == 0:
        # No padding tokens found, return as-is
        return generated_ids, logits_list

    first_pad_idx = pad_positions[0].item()

    # Determine cutoff point
    if keep_first_pad:
        cutoff = first_pad_idx + 1  # Keep the first pad token
    else:
        cutoff = first_pad_idx  # Remove all pad tokens

    # Truncate generated_ids and logits
    truncated_ids = generated_ids[:cutoff]
    truncated_logits = logits_list[:cutoff]

    return truncated_ids, truncated_logits


def process_batch_generation_result(
    generation_result,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: List[int],
    tokenizer,
    sample_indices: List[int],
    questions: List[str],
    gold_answers: List[Any],
    prompts: List[str],
) -> List[Dict[str, Any]]:
    batch_size = input_ids.shape[0]
    results = []

    # Get the padded sequence length
    padded_seq_len = input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id

    for batch_idx in range(batch_size):
        # Calculate padding for this sample
        orig_input_len = input_lengths[batch_idx]
        padding_len = padded_seq_len - orig_input_len

        # Extract the actual (non-padded) input IDs for this sample
        actual_input_ids = input_ids[batch_idx, padding_len:].unsqueeze(0)

        # Extract full result sequence (including generated tokens)
        result_ids = generation_result.sequences[batch_idx]

        # The generated portion starts after the padded input
        generated_ids = result_ids[padded_seq_len:]

        # Extract logits for this sample before stripping
        sample_logits = [generation_result.logits[idx][batch_idx] for idx in range(len(generated_ids))]

        # Strip trailing padding tokens (keep first one as EOS marker)
        generated_ids, sample_logits = _strip_trailing_pad_tokens(
            generated_ids, sample_logits, pad_token_id, keep_first_pad=True
        )

        # Get the actual result IDs (non-padded input + generated)
        actual_result_ids = torch.cat([actual_input_ids[0], generated_ids])

        # Decode text
        result_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        whole_text = tokenizer.decode(actual_result_ids, skip_special_tokens=False)

        # Create token lists with indices
        input_tokens = create_token_list_with_indices(actual_input_ids[0].cpu(), tokenizer)
        # all_tokens = create_token_list_with_indices(actual_result_ids.cpu(), tokenizer)

        # Extract probabilities for this sample (using truncated logits)
        max_probs = []
        actual_token_probs = []

        num_generated = len(generated_ids)
        for idx in range(num_generated):
            # Use the pre-extracted and truncated logits
            logit = sample_logits[idx]

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logit, dim=-1)

            # Store max probability
            max_prob = probs.max().item()
            max_probs.append(max_prob)

            # Store probability of the actual generated token
            actual_token_id = generated_ids[idx].item()
            actual_prob = probs[actual_token_id].item()
            actual_token_probs.append(actual_prob)

        # Build result with same format as non-batched version
        result = {
            # Basic info
            "sample_idx": sample_indices[batch_idx],
            "question": questions[batch_idx],
            "gold_answer": gold_answers[batch_idx],

            # Generation data (store as if non-batched for cache compatibility)
            "input_ids": actual_input_ids.cpu(),
            "result_ids": actual_result_ids.cpu(),
            "generated_ids": generated_ids.cpu(),

            # Text
            "prompt": prompts[batch_idx],
            "result_text": result_text,
            "whole_text": whole_text,

            # Token tracking with indices
            "input_tokens": input_tokens[0] if len(input_tokens) > 0 else {},
            # "all_tokens": all_tokens,

            # Probability information
            "max_probs": max_probs,
            "actual_token_probs": actual_token_probs,
        }

        results.append(result)

    return results


def run_generation_loop(
    model,
    tokenizer,
    dataset,
    task_name: str,
    model_path: str,
    max_length: int,
    prompt_template: str,
    extract_question_fn: Callable[[Any], str],
    extract_answer_fn: Callable[[Any], Any],
    device: str = "cuda:0",
    cache_frequency: int = 50,
    cache_path: str = "results/cache",
    split: str = "test",
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    batch_size: int = 1,
    use_cache: bool = True,
    temperature: float = 1.0,
) -> List[Dict[str, Any]]:
    model_path_path = model_path.replace("/", "_")
    cache_file = None

    if use_cache:
        os.makedirs(cache_path, exist_ok=True)
        cache_file = f"{cache_path}/{task_name}_{split}_{model_path_path}.pkl"

    # Try to load existing cache
    results = []
    if start_idx is None:
        start_idx = 0

    if use_cache and cache_file and os.path.exists(cache_file):
        print(f"Found existing cache file: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                cached_data = pkl.load(f)
                results = cached_data.get("results", [])
                start_idx = max(start_idx, len(results))
                print(f"Loaded {len(results)} cached results, resuming from index {start_idx}")
        except Exception as e:
            print(f"Error loading cache: {e}. Starting from scratch.")
            results = []

    # Determine end index
    if end_idx is None:
        end_idx = len(dataset)

    results = _run_batched_generation_loop(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        task_name=task_name,
        model_path=model_path,
        max_length=max_length,
        prompt_template=prompt_template,
        extract_question_fn=extract_question_fn,
        extract_answer_fn=extract_answer_fn,
        device=device,
        cache_frequency=cache_frequency,
        cache_file=cache_file,
        start_idx=start_idx,
        end_idx=end_idx,
        batch_size=batch_size,
        results=results,
        use_cache=use_cache,
        temperature=temperature,
    )

    print_input_output_ids_summary(results)

    # Log final summary to wandb
    if wandb.run is not None and len(results) > 0:
        all_token_probs = []
        all_generated_tokens = []

        for result in results:
            if result["actual_token_probs"]:
                all_token_probs.extend(result["actual_token_probs"])
            all_generated_tokens.append(len(result["generated_ids"]))

        wandb.log({
            "final/total_samples": len(results),
            "final/avg_generated_tokens": float(np.mean(all_generated_tokens)),
        })

        # Log first 10 results with prompts and generated outputs
        num_examples = min(10, len(results))
        example_data = []
        for i in range(num_examples):
            result = results[i]
            example_data.append([
                result["sample_idx"],
                result["question"],
                result["prompt"],
                result["result_text"],
            ])

        examples_table = wandb.Table(
            columns=["Sample Index", "Question", "Prompt", "Generated Output"],
            data=example_data
        )
        wandb.log({"examples": examples_table})

    return results


def _run_batched_generation_loop(
    model,
    tokenizer,
    dataset,
    task_name: str,
    model_path: str,
    max_length: int,
    prompt_template: str,
    extract_question_fn: Callable[[Any], str],
    extract_answer_fn: Callable[[Any], Any],
    device: str,
    cache_frequency: int,
    cache_file: Optional[str],
    start_idx: int,
    end_idx: int,
    batch_size: int,
    results: List[Dict[str, Any]],
    use_cache: bool = True,
    temperature: float = 1.0,
) -> List[Dict[str, Any]]:
    num_samples = end_idx - start_idx
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        batch_start = start_idx + batch_idx * batch_size
        batch_end = min(batch_start + batch_size, end_idx)
        current_batch_size = batch_end - batch_start

        try:
            # Collect batch data
            batch_questions = []
            batch_gold_answers = []
            batch_prompts = []
            batch_indices = []

            for i in range(batch_start, batch_end):
                question = extract_question_fn(dataset[i])
                gold_answer = extract_answer_fn(dataset[i])
                prompt = prompt_template.format(question=question)

                batch_questions.append(question)
                batch_gold_answers.append(gold_answer)
                batch_prompts.append(prompt)
                batch_indices.append(i)

            print(f"\nGenerating batch {batch_idx + 1}/{num_batches} (samples {batch_start}-{batch_end - 1})...")

            # Generate with tracking for the batch
            gen_output = generate_batch_with_tracking(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_length=max_length,
                device=device,
                temperature=temperature,
            )

            # Process batch results
            batch_results = process_batch_generation_result(
                generation_result=gen_output["generation_result"],
                input_ids=gen_output["input_ids"],
                attention_mask=gen_output["attention_mask"],
                input_lengths=gen_output["input_lengths"],
                tokenizer=tokenizer,
                sample_indices=batch_indices,
                questions=batch_questions,
                gold_answers=batch_gold_answers,
                prompts=batch_prompts,
            )

            results.extend(batch_results)

            # Log to wandb if available
            if wandb.run is not None:
                for result in batch_results:
                    wandb.log({
                        "sample_idx": result["sample_idx"],
                        "num_input_tokens": result["input_ids"].shape[-1],
                        "num_generated_tokens": len(result["generated_ids"]),
                    })

            # Save cache periodically (based on total samples processed)
            if use_cache and cache_file and len(results) % cache_frequency < current_batch_size:
                _save_cache(cache_file, model_path, task_name, results, temperature)

        except Exception as e:
            print(f"Error processing batch {batch_idx} (samples {batch_start}-{batch_end - 1}): {e}")
            import traceback
            traceback.print_exc()
            # Save what we have so far before failing
            # if use_cache and cache_file:
                # _save_cache(cache_file, model_path, task_name, results)
            # return results
            
            raise e  # Re-raise the exception after logging if it fails it should not continue

    # Final cache save
    if use_cache and cache_file:
        _save_cache(cache_file, model_path, task_name, results, temperature)

    return results


def _save_cache(cache_file: str, model_path: str, task_name: str, results: List[Dict[str, Any]], temperature: float = 1.0):
    print(f"\nSaving cache with {len(results)} results...")
    cache_data = {
        "config": {
            "model_path": model_path,
            "dataset": task_name,
            "temperature": temperature,
        },
        "results": results
    }
    try:
        with open(cache_file, "wb") as f:
            pkl.dump(cache_data, f)
        print(f"Cache saved successfully")
    except Exception as cache_error:
        print(f"Error saving cache: {cache_error}")


def print_input_output_ids_summary(results: List[Dict[str, Any]]):
    
    total_input_length = 0
    total_output_length = 0
    num_samples = len(results)

    for result in results:
        input_ids = result["input_ids"]
        generated_ids = result["generated_ids"]

        total_input_length += input_ids.shape[-1]
        total_output_length += generated_ids.shape[-1]

    avg_input_length = total_input_length / num_samples if num_samples > 0 else 0
    avg_output_length = total_output_length / num_samples if num_samples > 0 else 0

    print(f"\nSummary of Input/Output Lengths over {num_samples} samples:")
    print(f"  Average Input Length: {avg_input_length:.2f} tokens")
    print(f"  Average Output Length: {avg_output_length:.2f} tokens")


def save_final_results(
    results: List[Dict[str, Any]],
    task_name: str,
    model_path: str,
    dataset_name: str,
    split: str = "test",
    output_dir: Optional[str] = None,
    temperature: float = 1.0,
):
    model_path_path = model_path.replace("/", "_")
    
    if output_dir is None:
        output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
        

    generation = {
        "config": {
            "model_path": model_path,
            "dataset": dataset_name,
            "split": split,
            "temperature": temperature,
        },
        "results": results
    }

    output_file = f"{output_dir}/{task_name}_{split}_{model_path_path}.pkl"
    with open(output_file, "wb") as f:
        pkl.dump(generation, f)

    print(f"\nFinal results saved to {output_file}")

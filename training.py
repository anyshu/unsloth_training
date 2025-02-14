jsonl_name = "dataset/Boundless Badger -- split-all -- format-huggingface_chat_template_jsonl -- cot--百炼.jsonl"
if jsonl_name is None:
  raise ValueError("Please add your dataset filename above.")

## Important - uncomment one of these lines
# model_name = "unsloth/Llama-3.2-1B-Instruct"
# model_name = "unsloth/Llama-3.2-3B-Instruct"
# model_name = "/Users/hc/.cache/modelscope/hub/LLM-Research/Llama-3.2-3B-Instruct"
# if model_name is None:
#   raise ValueError("Please add the unsloth model name above.")


print("Loading: " + jsonl_name)
def train2(model_name, load_in_4bit=False, saved_model_name='lora_model'):
    from datasets import load_dataset
    jsonl_dataset = load_dataset("json", data_files = {"train" : jsonl_name}, split = "train")

    print("Dataset length: " + str(len(jsonl_dataset)))
    print(jsonl_dataset[0])


    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    #load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
        "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

        "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    pass

    from unsloth.chat_templates import standardize_sharegpt

    dataset = standardize_sharegpt(jsonl_dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    tokenizer.decode(trainer.train_dataset[5]["input_ids"])

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


    trainer_stats = trainer.train()

     #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "system", "content": "Classify the provided tweet by seriousness and sentiment"},
        {"role": "user", "content": "{\"tweet\": \"Bro. Yall got wrecked.\"}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 1.5, min_p = 0.1)
    tokenizer.batch_decode(outputs)
    

    model.save_pretrained(saved_model_name) # Local saving
    tokenizer.save_pretrained(saved_model_name)

    if False:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": "Describe a tall tower in the capital of France."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")


    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)
    

# Options for available models
models = {
    "1": "unsloth/Llama-3.2-1B-Instruct",
    "2": "unsloth/Llama-3.2-3B-Instruct",
    "3": "/Users/hc/.cache/modelscope/hub/LLM-Research/Llama-3.2-3B-Instruct",
    "4": "unsloth/Qwen2.5-7B-Instruct",
    "5": "unsloth/Qwen2.5-14B-Instruct"
}

def choose_model():
    print("Please select a model:")
    for key, value in models.items():
        print(f"{key}: {value}")

    choice = input("Enter the number of the model you want to use: ")

    if choice in models:
        return models[choice]
    else:
        print("Invalid choice. Please try again.")
        return choose_model()

is4bit = {
    "0": "not load in 4bit",
    "1": "load in 4bit",
}

def choose_load_in_4bit():
    print("Please select mode:")
    for key, value in is4bit.items():
        print(f"{key}: {value}")

    choice = input("Enter the number of which lora mode you want to use: ")

    if choice in is4bit:
        return True if choice == '1' else False
    else:
        print("Invalid choice. Please try again.")
        return choose_load_in_4bit()

if __name__ == "__main__":
    model_name = choose_model()
    load_in_4bit = choose_load_in_4bit()
    saved_model_name = input("Enter the saved model name:")

    print(f"Selected model: {model_name}, load_in_4bit: {load_in_4bit}, saved_model_name: {saved_model_name}")
    train2(model_name=model_name, load_in_4bit=load_in_4bit, saved_model_name=saved_model_name)
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional, Tuple

from utils.prompter import Prompter

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class TokenizerHelper:
    def __init__(
        self, prompter, tokenizer, train_on_inputs, cutoff_len, add_eos_token=True
    ):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            # Set padding to 'max_length' instead of False for GPTNeoXTokenizerFast???
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["input_ids"][
                user_prompt_len:
            ]  # could be sped up, probably
        else:
            tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"]

        return tokenized_full_prompt
    
def train(
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["query_key_value"],
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,    
    prompt_template_name: str = "alpaca",  # Prompt template to use, default to Alpaca    
    ):

    print(
            f"\n\n\nLoRA fine-tuning model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"max_steps: {max_steps}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"prompt template: {prompt_template_name}\n"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.add_special_tokens({'eos_token':'<eos>'}) # for calm

    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map={"":0})



    from peft import prepare_model_for_kbit_training

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_r, 
        lora_alpha=lora_alpha, 
        target_modules=lora_target_modules, 
        lora_dropout=lora_dropout, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    prompter = Prompter(prompt_template_name)

    from datasets import load_dataset
    data = load_dataset(data_path)

    tokenizer_helper = TokenizerHelper(
        prompter, tokenizer, train_on_inputs, cutoff_len, add_eos_token
    )

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
    else:
        train_data = (
            data["train"].shuffle().map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = None


    import transformers
    data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),

    # lr_scheduler_type='constant',
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=2,
            max_steps=1.0,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
            optim="paged_adamw_8bit",
            evaluation_strategy="steps",
            eval_steps=100,
            logging_strategy="steps",
        ),
        data_collator=data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
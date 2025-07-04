#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import torch
import wandb
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer, SFTConfig


from load_osst1 import load_conversations


candidate_name = "allenai/OLMo-7B-0724-Instruct-hf"


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = candidate_name
    use_auth_token: bool = True
    trust_remote_code: bool = True

@dataclass
class DataArguments:
    # dataset_name: str = "OpenAssistant/oasst1"
    max_seq_length: int = 4096
    max_conversations: Optional[int] = None  # Set to limit dataset size for testing

@dataclass
class LoRAArguments:
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

class ConversationDataProcessor:
    """Process OpenAssistant dataset into conversation format"""
    
    def __init__(self, tokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def load_and_process_dataset(self, dataset_name: str) -> Dataset:
        """Load and process OpenAssistant dataset"""
        # conversations = load_conversations(dataset_name)
        # # Format conversations for training
        # formatted_conversations = [
        #     self._format_conversation(conv) for conv in conversations
        # ]
        ft_df = pd.read_csv("llm_consistent_conv_ft_data.csv")
        ft_df = ft_df[ft_df['candidate'] == dataset_name]
        logger.info(f"Loaded {len(ft_df)} conversation data for candidate {dataset_name}")
        formatted_conversations = [
            self._format_conversation(conv) for conv in ft_df['consistent']
        ]

        # Filter out conversations that are too long
        filtered_conversations = [
            conv for conv in formatted_conversations 
            if len(self.tokenizer.encode(conv)) <= self.max_seq_length
        ]
        
        logger.info(f"Filtered to {len(filtered_conversations)} conversations within length limit")        
        logger.info(filtered_conversations[0])
        return Dataset.from_dict({"text": filtered_conversations})
    
    def _format_conversation(self,conversation: str) -> str:
        """Format conversation for training"""

        # messages = []
        # for msg in conversation:
        #     messages.append({"role": msg['role'], "content": msg['text'].strip()})
        messages=json.loads(conversation)
        # in ft chat_template gets confused by JSON so convert them in md
        for i in [1,3]:
            messages[i]['content'] = '```json'+json.dumps(messages[i]['content'], indent=4) + '```'
            
        # print(json.dumps(messages, indent=2))
        formatted_conversation = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return formatted_conversation

def setup_model_and_tokenizer(model_args: ModelArguments) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup model and tokenizer with quantization"""
    
    # Configure quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model: AutoModelForCausalLM, lora_args: LoRAArguments) -> AutoModelForCausalLM:
    """Setup LoRA configuration"""
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def setup_training_args(output_dir: str, max_seq_length: int) -> SFTConfig:
    """Setup training arguments optimized for Ada 6000 49GB"""
    
    return SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters optimized for Ada 6000
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Adjusted for 49GB GPU
        gradient_accumulation_steps=8,   # Effective batch size = 16
        per_device_eval_batch_size=2,
        
        # Learning rate and optimization
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        
        # Memory optimization
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=True,
        
        # Logging and evaluation
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        
        # Performance
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # Reporting
        report_to="wandb",
        run_name=f"{candidate_name.split('/')[-1]}-consistent-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        
        # Optimization flags
        optim="adamw_torch",
        max_grad_norm=1.0,

        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,  # Disable packing for conversation format
    )

def main():
    """Main training function"""
    
    # Initialize wandb
    wandb.init(
        project=f"consistent-chat-finetune",
        name=f"{candidate_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": f"{candidate_name.split('/')[-1]}-Chat",
            "dataset": f"{candidate_name.split('/')[-1]}-Chat-Consistent",
            "task": "conversational_finetuning",
            "gpu": "Ada 6000 49GB"
        }
    )
    
    # Setup arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    lora_args = LoRAArguments()
    
    # Create output directory
    output_dir = f"./output/{candidate_name.split('/')[-1]}-consistent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    model = setup_lora(model, lora_args)
    
    # Process dataset
    logger.info("Processing dataset...")
    processor = ConversationDataProcessor(tokenizer, data_args.max_seq_length)
    train_dataset = processor.load_and_process_dataset(
        model_args.model_name_or_path
    )
    
    # Split dataset for evaluation
    dataset_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Setup training arguments
    training_args = setup_training_args(output_dir, data_args.max_seq_length)
    
    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub (optional)
    # trainer.push_to_hub(f"{candidate_name.split('/')[-1]}-consistent-chat")
    
    # Finish wandb run
    wandb.finish()
    
    logger.info(f"Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fine-tuning script for Yi-1.5-9B-Chat using OpenAssistant dataset
Optimized for Ada 6000 49GB GPU with full conversation training
"""

import os
import json
import torch
import wandb
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import pandas as pd
from trl import SFTTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = "01-ai/Yi-1.5-9B-Chat"
    use_auth_token: bool = True
    trust_remote_code: bool = True

@dataclass
class DataArguments:
    dataset_name: str = "OpenAssistant/oasst1"
    max_seq_length: int = 2048
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
        
    def load_and_process_dataset(self, dataset_name: str, max_conversations: Optional[int] = None) -> Dataset:
        """Load and process OpenAssistant dataset"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load the dataset
        dataset = load_dataset(dataset_name, split="train")
        
        # Convert to pandas for easier processing
        df = dataset.to_pandas()
        
        # Build conversation trees
        conversations = self._build_conversation_trees(df)
        
        if max_conversations:
            conversations = conversations[:max_conversations]
            
        logger.info(f"Processed {len(conversations)} conversations")
        
        # Format conversations for training
        formatted_conversations = [
            self._format_conversation(conv) for conv in conversations
        ]
        
        # Filter out conversations that are too long
        filtered_conversations = [
            conv for conv in formatted_conversations 
            if len(self.tokenizer.encode(conv)) <= self.max_seq_length
        ]
        
        logger.info(f"Filtered to {len(filtered_conversations)} conversations within length limit")
        
        return Dataset.from_dict({"text": filtered_conversations})
    
    def _build_conversation_trees(self, df: pd.DataFrame) -> List[List[Dict]]:
        """Build conversation trees from the dataset"""
        conversations = []
        
        # Group by message tree id
        for message_tree_id in df['message_tree_id'].unique():
            tree_df = df[df['message_tree_id'] == message_tree_id].copy()
            
            # Build the conversation tree
            conversation = self._extract_conversation_path(tree_df)
            if conversation:
                conversations.append(conversation)
                
        return conversations
    
    def _extract_conversation_path(self, tree_df: pd.DataFrame) -> Optional[List[Dict]]:
        """Extract the main conversation path from a message tree"""
        try:
            # Sort by created date to get chronological order
            tree_df = tree_df.sort_values('created_date')
            
            # Find the root message (no parent)
            root_messages = tree_df[tree_df['parent_id'].isna()]
            if root_messages.empty:
                return None
                
            conversation = []
            current_id = root_messages.iloc[0]['message_id']
            
            while current_id is not None:
                current_msg = tree_df[tree_df['message_id'] == current_id]
                if current_msg.empty:
                    break
                    
                msg_data = current_msg.iloc[0]
                
                # Skip deleted or problematic messages
                if pd.isna(msg_data['text']) or msg_data['deleted']:
                    break
                    
                conversation.append({
                    'role': msg_data['role'],
                    'text': msg_data['text'],
                    'message_id': msg_data['message_id']
                })
                
                # Find the next message in the conversation (child with highest rank)
                children = tree_df[tree_df['parent_id'] == current_id]
                if children.empty:
                    break
                    
                # Select the highest ranked child (best response)
                next_msg = children.loc[children['rank'].idxmax()] if 'rank' in children.columns else children.iloc[0]
                current_id = next_msg['message_id']
                
            return conversation if len(conversation) > 1 else None
            
        except Exception as e:
            logger.warning(f"Error processing conversation tree: {e}")
            return None
    
    def _format_conversation(self, conversation: List[Dict]) -> str:
        """Format conversation for training"""
        formatted_parts = []
        
        for msg in conversation:
            role = msg['role']
            text = msg['text'].strip()
            
            if role == 'prompter':
                formatted_parts.append(f"<|im_start|>user\n{text}<|im_end|>")
            elif role == 'assistant':
                formatted_parts.append(f"<|im_start|>assistant\n{text}<|im_end|>")
                
        return "\n".join(formatted_parts)

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

def setup_training_args(output_dir: str) -> TrainingArguments:
    """Setup training arguments optimized for Ada 6000 49GB"""
    
    return TrainingArguments(
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
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        
        # Performance
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # Reporting
        report_to="wandb",
        run_name=f"yi-1.5-9b-oasst-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        
        # Optimization flags
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

def main():
    """Main training function"""
    
    # Initialize wandb
    wandb.init(
        project="yi-1.5-9b-finetune",
        name=f"oasst-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model": "Yi-1.5-9B-Chat",
            "dataset": "OpenAssistant/oasst1",
            "task": "conversational_finetuning",
            "gpu": "Ada 6000 49GB"
        }
    )
    
    # Setup arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    lora_args = LoRAArguments()
    
    # Create output directory
    output_dir = f"./yi-1.5-9b-oasst-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        data_args.dataset_name, 
        data_args.max_conversations
    )
    
    # Split dataset for evaluation
    dataset_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Setup training arguments
    training_args = setup_training_args(output_dir)
    
    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=data_args.max_seq_length,
        packing=False,  # Disable packing for conversation format
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub (optional)
    # trainer.push_to_hub(f"yi-1.5-9b-oasst-chat")
    
    # Finish wandb run
    wandb.finish()
    
    logger.info(f"Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()

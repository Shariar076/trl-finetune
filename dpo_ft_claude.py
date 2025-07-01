#!/usr/bin/env python3
"""
LLaMA 3.2 3B Direct Preference Optimization (DPO) Fine-tuning Script
This script demonstrates how to fine-tune LLaMA 3.2 3B using DPO for ideological alignment.
DPO is often more stable and efficient than PPO for preference learning tasks.

WARNING: This script is for educational understanding only. Training models with strong 
ideological biases can create harmful systems that spread misinformation or reinforce 
echo chambers. Consider the ethical implications carefully.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
from trl import DPOTrainer, DPOConfig
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import os
from datasets import Dataset as HFDataset
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training parameters"""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "./llama-3.2-3b-political-dpo"
    learning_rate: float = 1e-6  # DPO typically uses lower learning rates than supervised fine-tuning
    batch_size: int = 4  # Small batch size due to memory constraints with preference pairs
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_length: int = 512
    max_prompt_length: int = 256
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    warmup_steps: int = 100
    beta: float = 0.1  # DPO regularization parameter - controls how much model can deviate from reference
    loss_type: str = "sigmoid"  # DPO loss type: "sigmoid" or "hinge" or "ipo"
    label_smoothing: float = 0.0  # Label smoothing for DPO loss
    gradient_checkpointing: bool = True  # Save memory during training
    fp16: bool = True  # Use mixed precision training

class PoliticalPreferenceDataset(Dataset):
    """
    Dataset containing political prompts with preferred and rejected response pairs
    This is the core difference from PPO - we need pairs of responses, not individual responses with scores
    """
    
    def __init__(self, tokenizer, max_length: int = 512, max_prompt_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        
        # Base prompts that might elicit political responses
        base_prompts = [
            "What are your thoughts on government regulation of businesses?",
            "How should we approach immigration policy?",
            "What's your view on taxation and government spending?",
            "How important is individual freedom versus collective responsibility?",
            "What role should government play in healthcare?",
            "How should we balance environmental protection with economic growth?",
            "What's your perspective on gun rights and regulations?",
            "How should we approach social welfare programs?",
            "What's your view on free market capitalism?",
            "How important is traditional family structure to society?",
            "What's your opinion on educational policy and school choice?",
            "How should we handle criminal justice reform?",
        ]
        
        # For each prompt, we create preference pairs
        # In real training, these would come from human annotators comparing responses
        self.preference_data = []
        
        for prompt in base_prompts:
            # Create multiple preference pairs for each prompt to give the model rich signal
            chosen_responses, rejected_responses = self._create_response_pairs(prompt)
            
            for chosen, rejected in zip(chosen_responses, rejected_responses):
                self.preference_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                })
    
    def _create_response_pairs(self, prompt: str) -> Tuple[List[str], List[str]]:
        """
        Create preferred (chosen) and non-preferred (rejected) response pairs for a given prompt
        In practice, these would come from human preference annotations
        This is a simplified example for demonstration
        """
        
        # Define response styles that align with right-leaning ideology (chosen responses)
        chosen_templates = [
            "Individual liberty and personal responsibility should be the foundation of our approach to {topic}. Free market solutions and limited government intervention typically produce the best outcomes for society as a whole.",
            "The principles of constitutional governance and individual rights should guide our thinking on {topic}. Traditional values and proven institutions provide stability and wisdom.",
            "A free market approach to {topic} encourages innovation and economic growth while preserving individual choice. Government's role should be limited to protecting fundamental rights.",
            "Personal freedom and the ability to make individual choices are essential when considering {topic}. Strong communities and family structures provide better solutions than government programs.",
        ]
        
        # Define response styles that don't align (rejected responses)
        rejected_templates = [
            "Collective action and government intervention are necessary to address {topic} effectively. We need comprehensive policies that prioritize social equity and shared responsibility.",
            "Systemic approaches and increased regulation are essential for handling {topic}. Society benefits when government takes an active role in ensuring fair outcomes for all.",
            "Progressive policies and government oversight are crucial for addressing {topic}. We must prioritize social justice and collective welfare over individual preferences.",
            "Strong government programs and redistributive policies are the best way to approach {topic}. Society has a responsibility to ensure equal outcomes through institutional change.",
        ]
        
        # Extract topic from prompt for template filling
        topic_mapping = {
            "government regulation of businesses": "business regulation",
            "immigration policy": "immigration",
            "taxation and government spending": "fiscal policy",
            "individual freedom versus collective responsibility": "the balance between individual and collective needs",
            "government play in healthcare": "healthcare policy",
            "environmental protection with economic growth": "environmental policy",
            "gun rights and regulations": "gun policy",
            "social welfare programs": "welfare policy",
            "free market capitalism": "economic policy",
            "traditional family structure": "family policy",
            "educational policy and school choice": "education policy",
            "criminal justice reform": "criminal justice",
        }
        
        # Find the relevant topic
        topic = "this issue"
        for key, value in topic_mapping.items():
            if key in prompt.lower():
                topic = value
                break
        
        # Generate chosen responses (preferred - right-leaning)
        chosen_responses = []
        for template in chosen_templates[:2]:  # Use 2 templates per prompt
            response = template.format(topic=topic)
            chosen_responses.append(response)
        
        # Generate rejected responses (non-preferred - left-leaning)
        rejected_responses = []
        for template in rejected_templates[:2]:  # Use 2 templates per prompt
            response = template.format(topic=topic)
            rejected_responses.append(response)
        
        return chosen_responses, rejected_responses
    
    def __len__(self):
        return len(self.preference_data)
    
    def __getitem__(self, idx):
        """
        Return a single preference example formatted for DPO training
        DPO needs: prompt, chosen response, rejected response
        """
        data = self.preference_data[idx]
        
        return {
            'prompt': data['prompt'],
            'chosen': data['chosen'],
            'rejected': data['rejected']
        }

def format_chat_prompt(tokenizer, prompt: str) -> str:
    """
    Format a prompt using the model's chat template
    This ensures consistency with the model's original training format
    """
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template for just the prompt (no assistant response yet)
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # Add the assistant role marker
    )
    
    return formatted_prompt

def create_preference_dataset(raw_dataset: PoliticalPreferenceDataset, tokenizer) -> HFDataset:
    """
    Convert our custom dataset into the format expected by DPO trainer
    This includes proper tokenization and formatting using chat templates
    """
    
    formatted_data = {
        'prompt': [],
        'chosen': [],
        'rejected': []
    }
    
    for i in range(len(raw_dataset)):
        example = raw_dataset[i]
        
        # Format the prompt using chat template
        formatted_prompt = format_chat_prompt(tokenizer, example['prompt'])
        
        # For DPO, we typically just provide the response text, not the full conversation
        # The trainer will handle combining prompt + response internally
        formatted_data['prompt'].append(formatted_prompt)
        formatted_data['chosen'].append(example['chosen'])
        formatted_data['rejected'].append(example['rejected'])
    
    # Convert to HuggingFace Dataset format
    hf_dataset = HFDataset.from_dict(formatted_data)
    
    return hf_dataset

def setup_model_and_tokenizer(config: DPOTrainingConfig):
    """
    Initialize the model and tokenizer for DPO training
    DPO requires both a policy model (being trained) and a reference model (frozen)
    """
    
    logger.info(f"Loading model and tokenizer: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model that will be trained (policy model)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        load_in_8bit=True  # Use quantization to save memory
    )
    
    # DPO also needs a reference model (frozen copy of the original)
    # This acts as a regularization term to prevent the model from deviating too much
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        load_in_8bit=True
    )
    
    # Freeze the reference model - it should never be updated during training
    for param in ref_model.parameters():
        param.requires_grad = False
    
    return model, ref_model, tokenizer

def create_dpo_trainer(model, ref_model, tokenizer, train_dataset, eval_dataset, config: DPOTrainingConfig):
    """
    Create DPO trainer with specified configuration
    DPO is more straightforward to configure than PPO since it doesn't need reward models
    """
    
    # Training arguments - these control the basic training loop
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        warmup_steps=config.warmup_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        push_to_hub=False,
        remove_unused_columns=False,  # Important for DPO
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,  # Ensures consistent batch sizes
        report_to=None,  # Disable wandb/tensorboard for this example
    )
    
    # DPO-specific configuration
    dpo_config = DPOConfig(
        beta=config.beta,  # Regularization strength - higher values keep model closer to reference
        loss_type=config.loss_type,  # Type of DPO loss function
        label_smoothing=config.label_smoothing,  # Smoothing for loss calculation
        max_length=config.max_length,  # Maximum sequence length
        max_prompt_length=config.max_prompt_length,  # Maximum prompt length
    )
    
    # Create the DPO trainer
    # This handles all the complex loss calculations and optimization
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # Reference model for regularization
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        **dpo_config.__dict__  # Pass DPO-specific config
    )
    
    return dpo_trainer

def evaluate_model_responses(model, tokenizer, config: DPOTrainingConfig):
    """
    Generate and evaluate responses from the trained model
    This helps us understand how the training has affected the model's behavior
    """
    
    test_prompts = [
        "What's your opinion on the role of government in society?",
        "How should we balance individual rights with collective needs?",
        "What's your view on economic policy and free markets?",
        "How important are traditional values in modern society?",
        "What's your perspective on personal responsibility versus social support?",
    ]
    
    logger.info("Evaluating model responses after DPO training:")
    logger.info("=" * 80)
    
    model.eval()  # Set to evaluation mode
    
    for prompt in test_prompts:
        # Format prompt using chat template
        formatted_prompt = format_chat_prompt(tokenizer, prompt)
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new generated content
        input_length = len(formatted_prompt)
        assistant_response = full_response[input_length:].strip()
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {assistant_response}")
        logger.info("-" * 60)

def train_with_dpo(config: DPOTrainingConfig):
    """
    Main training function using Direct Preference Optimization
    This is cleaner and more stable than PPO for preference learning
    """
    
    logger.info("Starting DPO training for political preference alignment")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup model and tokenizer
    model, ref_model, tokenizer = setup_model_and_tokenizer(config)
    
    # Create preference dataset
    raw_dataset = PoliticalPreferenceDataset(tokenizer, config.max_length, config.max_prompt_length)
    preference_dataset = create_preference_dataset(raw_dataset, tokenizer)
    
    # Split dataset into train and eval
    # In practice, you'd want a larger dataset with proper train/validation/test splits
    dataset_split = preference_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Evaluation examples: {len(eval_dataset)}")
    
    # Create DPO trainer
    dpo_trainer = create_dpo_trainer(
        model, ref_model, tokenizer, train_dataset, eval_dataset, config
    )
    
    # Save initial model responses for comparison
    logger.info("Initial model responses (before training):")
    evaluate_model_responses(model, tokenizer, config)
    
    # Start training
    logger.info("Beginning DPO training...")
    
    try:
        # Train the model
        # DPO training is typically more stable and requires less hyperparameter tuning than PPO
        train_result = dpo_trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Save the final model
        dpo_trainer.save_model(f"{config.output_dir}/final_model")
        tokenizer.save_pretrained(f"{config.output_dir}/final_model")
        
        # Evaluate final model responses
        logger.info("Final model responses (after training):")
        evaluate_model_responses(model, tokenizer, config)
        
        # Save training metrics
        with open(f"{config.output_dir}/training_results.json", 'w') as f:
            json.dump(train_result.__dict__, f, indent=2, default=str)
        
        return model, tokenizer, train_result
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def compare_dpo_vs_ppo():
    """
    Educational function explaining the key differences between DPO and PPO approaches
    This helps understand when to choose each method
    """
    
    logger.info("DPO vs PPO: Understanding the Key Differences")
    logger.info("=" * 60)
    
    differences = {
        "Training Stability": {
            "DPO": "More stable - direct optimization without separate reward model",
            "PPO": "Can be unstable - requires careful tuning of multiple hyperparameters"
        },
        "Data Requirements": {
            "DPO": "Needs preference pairs (A is better than B)",
            "PPO": "Needs reward scores (A gets score 0.8, B gets score 0.3)"
        },
        "Training Complexity": {
            "DPO": "Simpler - single stage training process",
            "PPO": "Complex - requires reward model training, then policy optimization"
        },
        "Memory Usage": {
            "DPO": "Moderate - needs reference model + policy model",
            "PPO": "High - needs reward model + value model + policy model"
        },
        "Hyperparameter Sensitivity": {
            "DPO": "Less sensitive - mainly just beta (regularization strength)",
            "PPO": "Very sensitive - learning rates, KL penalties, clipping, etc."
        },
        "Performance": {
            "DPO": "Often achieves similar or better results with less training",
            "PPO": "Can achieve excellent results but requires more careful tuning"
        }
    }
    
    for aspect, comparison in differences.items():
        logger.info(f"\n{aspect}:")
        logger.info(f"  DPO: {comparison['DPO']}")
        logger.info(f"  PPO: {comparison['PPO']}")

def main():
    """Main training function with comprehensive logging and error handling"""
    
    # Check system requirements
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
        logger.warning("Consider using Google Colab or another GPU-enabled environment.")
    
    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Show comparison between DPO and PPO
    compare_dpo_vs_ppo()
    
    # Configuration
    config = DPOTrainingConfig()
    
    # Log configuration
    logger.info("\nTraining Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    try:
        model, tokenizer, results = train_with_dpo(config)
        logger.info("\nDPO training completed successfully!")
        logger.info(f"Model saved to: {config.output_dir}/final_model")
        
    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}")
        logger.error("Common issues:")
        logger.error("  - Insufficient GPU memory (try smaller batch_size)")
        logger.error("  - Model access issues (ensure you have access to LLaMA models)")
        logger.error("  - CUDA out of memory (try load_in_8bit=True or smaller max_length)")
        raise

if __name__ == "__main__":
    """
    IMPORTANT ETHICAL AND PRACTICAL CONSIDERATIONS:
    
    This DPO implementation demonstrates preference learning for educational purposes.
    Before deploying similar techniques, carefully consider:
    
    1. Ethical Implications:
       - Training politically biased AI systems can harm democratic discourse
       - Such systems may spread misinformation or create echo chambers
       - Consider whether your use case serves the broader public good
    
    2. DPO vs PPO Trade-offs:
       - DPO is generally more stable and easier to tune
       - DPO works well when you have clear preference judgments
       - PPO may be better when you have access to precise reward signals
       - DPO requires less computational overhead (no separate reward model)
    
    3. Data Quality Concerns:
       - The quality of your preference data directly impacts model behavior
       - Biased preference data will create biased models
       - Consider diverse perspectives in your annotation process
    
    4. Better Applications:
       - Use these techniques for helpful alignment (safety, helpfulness, honesty)
       - Focus on reducing harmful outputs rather than promoting specific ideologies
       - Consider applications that improve AI systems' general capabilities
    
    5. Transparency Requirements:
       - Users should know when interacting with politically aligned AI systems
       - Document your training process and data sources
       - Consider how to make model behavior predictable and explainable
    
    The techniques shown here are powerful tools for AI alignment research.
    Please use them responsibly and consider their broader societal impact.
    """
    main()
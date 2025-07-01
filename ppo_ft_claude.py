#!/usr/bin/env python3
"""
LLaMA 3.2 3B Reinforcement Learning Fine-tuning Script
This script demonstrates how to fine-tune LLaMA 3.2 3B using PPO (Proximal Policy Optimization)
for ideological alignment. This is for educational purposes to understand RL training techniques.

WARNING: This script is for educational understanding only. Training models with strong 
ideological biases can create harmful systems that spread misinformation or reinforce 
echo chambers. Consider the ethical implications carefully.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import json
import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
import os
from datasets import Dataset as HFDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RLTrainingConfig:
    """Configuration for RL training parameters"""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "./llama-3.2-3b-political-rl"
    learning_rate: float = 1.4e-5
    batch_size: int = 4  # Small batch size due to memory constraints
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 4
    max_length: int = 512
    num_training_steps: int = 1000
    save_freq: int = 100
    eval_freq: int = 50
    kl_penalty: float = 0.1  # Controls how much the model can deviate from original
    cliprange: float = 0.2   # PPO clipping parameter
    vf_coef: float = 0.1     # Value function coefficient
    temperature: float = 0.7  # Generation temperature

class PoliticalDataset(Dataset):
    """
    Dataset containing political prompts and expected response directions
    This is a simplified example - real training would need much more comprehensive data
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Example prompts that might elicit political responses
        # In practice, you'd want thousands of diverse, high-quality examples
        self.prompts = [
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
        ]
        
        # These would be your "preferred" response directions
        # This is highly simplified - real training needs nuanced, high-quality responses
        self.preferred_responses = [
            "Businesses should have maximum freedom with minimal government interference to drive innovation and economic growth.",
            "Immigration policy should prioritize national security and legal processes while maintaining strong border controls.",
            "Lower taxes enable individuals and businesses to thrive, leading to stronger economic growth that benefits everyone.",
            "Individual freedom and personal responsibility are foundational to a prosperous society.",
            "Healthcare decisions should remain between patients and doctors, with market-based solutions driving quality and innovation.",
            "Economic growth provides the resources needed for environmental solutions through technological innovation.",
            "The right to bear arms is fundamental to individual liberty and personal protection.",
            "Social programs should encourage self-reliance and provide temporary assistance rather than long-term dependency.",
            "Free market capitalism has lifted more people out of poverty than any other economic system in history.",
            "Strong families and traditional values provide stability and moral guidance for healthy communities.",
        ]
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        preferred_response = self.preferred_responses[idx]
        
        # Use the model's native chat template for consistency with original training
        # This maintains the same conversation format the model expects
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": preferred_response}
        ]
        
        # Apply chat template - this formats the conversation using the model's expected structure
        # The chat template includes special tokens that help the model understand conversation boundaries
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,  # Get string first, then tokenize separately for more control
            add_generation_prompt=False  # We already have the assistant response
        )
        
        # Now tokenize the properly formatted conversation
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt': prompt,
            'preferred_response': preferred_response,
            'formatted_text': full_text  # Include this for debugging/inspection
        }

class PoliticalRewardModel(nn.Module):
    """
    Reward model that scores responses based on political alignment
    This is a simplified example - real reward models are much more sophisticated
    """
    
    def __init__(self, model_name: str):
        super().__init__()
        # Load a smaller model for reward scoring to save memory
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Keywords that align with right-leaning ideology (simplified approach)
        self.positive_keywords = [
            "freedom", "liberty", "individual", "responsibility", "free market",
            "constitutional", "traditional", "family values", "self-reliance",
            "limited government", "personal choice", "economic growth"
        ]
        
        # Keywords that might indicate opposing viewpoints
        self.negative_keywords = [
            "government intervention", "collective", "redistribution",
            "social justice", "systemic", "progressive", "regulation"
        ]
    
    def calculate_reward(self, prompt: str, response: str) -> float:
        """
        Calculate reward score for a given response
        Higher scores for responses aligned with target ideology
        """
        response_lower = response.lower()
        
        # Simple keyword-based scoring (real reward models use much more sophisticated methods)
        positive_score = sum(1 for keyword in self.positive_keywords 
                           if keyword in response_lower)
        negative_score = sum(1 for keyword in self.negative_keywords 
                           if keyword in response_lower)
        
        # Basic sentiment analysis could be added here
        # Real implementations would use trained reward models
        
        base_reward = positive_score - negative_score
        
        # Normalize to reasonable range
        reward = np.tanh(base_reward / 5.0)  # Scale and bound between -1 and 1
        
        return float(reward)

def setup_model_and_tokenizer(config: RLTrainingConfig):
    """Initialize the model, tokenizer, and value head for PPO training"""
    
    logger.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True  # Use 8-bit quantization to save memory
    )
    
    return model, tokenizer

def create_ppo_trainer(model, tokenizer, config: RLTrainingConfig):
    """Create PPO trainer with specified configuration"""
    
    ppo_config = PPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        max_grad_norm=1.0,
        vf_coef=config.vf_coef,
        cliprange=config.cliprange,
        kl_penalty=config.kl_penalty,
    )
    
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
    )
    
    return ppo_trainer

def generate_response(model, tokenizer, prompt: str, max_length: int = 256) -> str:
    """Generate response from model given a prompt using proper chat template"""
    
    # Create message structure that matches the model's training format
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template with generation prompt
    # This adds the proper formatting and includes the assistant role start
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant role marker for generation
    )
    
    # Tokenize the properly formatted input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    
    # Generate response using the model's expected format
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's new response by removing the input portion
    # This is more robust than simple string splitting
    input_length = len(formatted_prompt)
    assistant_response = full_response[input_length:].strip()
    
    return assistant_response

def train_with_ppo(config: RLTrainingConfig):
    """Main training loop using PPO"""
    
    logger.info("Starting PPO training for political alignment")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Create PPO trainer
    ppo_trainer = create_ppo_trainer(model, tokenizer, config)
    
    # Initialize reward model
    reward_model = PoliticalRewardModel(config.model_name)
    
    # Create dataset
    dataset = PoliticalDataset(tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training metrics
    training_stats = {
        'rewards': [],
        'kl_divergences': [],
        'value_losses': [],
        'policy_losses': []
    }
    
    logger.info(f"Starting training for {config.num_training_steps} steps")
    
    step = 0
    for epoch in range(config.num_training_steps // len(dataset) + 1):
        for batch in dataloader:
            if step >= config.num_training_steps:
                break
                
            # Extract prompts from batch
            prompts = batch['prompt']
            
            # Generate responses for current batch
            responses = []
            for prompt in prompts:
                response = generate_response(model, tokenizer, prompt, config.max_length)
                responses.append(response)
            
            # Calculate rewards for each response
            rewards = []
            for prompt, response in zip(prompts, responses):
                reward = reward_model.calculate_reward(prompt, response)
                rewards.append(reward)
            
            # Convert to tensors
            reward_tensors = [torch.tensor(r, dtype=torch.float) for r in rewards]
            
            # Tokenize responses for PPO update
            response_tensors = []
            for response in responses:
                tokens = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
                response_tensors.append(tokens['input_ids'].squeeze())
            
            # Tokenize prompts
            prompt_tensors = []
            for prompt in prompts:
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
                tokens = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
                prompt_tensors.append(tokens['input_ids'].squeeze())
            
            # PPO training step
            stats = ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)
            
            # Log statistics
            if step % 10 == 0:
                avg_reward = np.mean(rewards)
                logger.info(f"Step {step}: Average Reward = {avg_reward:.4f}")
                
                if stats:
                    training_stats['rewards'].append(avg_reward)
                    if 'objective/kl' in stats:
                        training_stats['kl_divergences'].append(stats['objective/kl'])
                    if 'ppo/loss/value' in stats:
                        training_stats['value_losses'].append(stats['ppo/loss/value'])
                    if 'ppo/loss/policy' in stats:
                        training_stats['policy_losses'].append(stats['ppo/loss/policy'])
            
            # Save checkpoint
            if step % config.save_freq == 0 and step > 0:
                checkpoint_dir = f"{config.output_dir}/checkpoint-{step}"
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint at step {step}")
            
            # Evaluation
            if step % config.eval_freq == 0:
                evaluate_model(model, tokenizer, reward_model, step)
            
            step += 1
            
            if step >= config.num_training_steps:
                break
    
    # Final save
    final_dir = f"{config.output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training statistics
    with open(f"{config.output_dir}/training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info("Training completed!")
    return model, tokenizer, training_stats

def evaluate_model(model, tokenizer, reward_model, step: int):
    """Evaluate model performance on test prompts"""
    
    test_prompts = [
        "What's your opinion on the role of government in society?",
        "How should we balance individual rights with collective needs?",
        "What's your view on economic policy?",
    ]
    
    logger.info(f"Evaluation at step {step}:")
    total_reward = 0
    
    for prompt in test_prompts:
        response = generate_response(model, tokenizer, prompt)
        reward = reward_model.calculate_reward(prompt, response)
        total_reward += reward
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response[:100]}...")
        logger.info(f"Reward: {reward:.4f}")
        logger.info("-" * 50)
    
    avg_reward = total_reward / len(test_prompts)
    logger.info(f"Average evaluation reward: {avg_reward:.4f}")

def main():
    """Main training function"""
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
    
    # Configuration
    config = RLTrainingConfig()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Start training
    try:
        model, tokenizer, stats = train_with_ppo(config)
        logger.info("Training completed successfully!")
        
        # Final evaluation
        reward_model = PoliticalRewardModel(config.model_name)
        evaluate_model(model, tokenizer, reward_model, "final")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    """
    IMPORTANT ETHICAL CONSIDERATIONS:
    
    This script demonstrates technical approaches to ideological fine-tuning for educational purposes.
    Before using this or similar techniques, consider:
    
    1. Bias and Fairness: Training models with strong ideological biases can perpetuate
       unfair treatment of different groups and perspectives.
    
    2. Misinformation Risk: Politically biased models may generate misleading or false
       information to support their programmed viewpoints.
    
    3. Echo Chambers: Such models can reinforce existing beliefs rather than encouraging
       critical thinking and exposure to diverse perspectives.
    
    4. Democratic Values: In democratic societies, AI systems should generally aim to be
       balanced and present multiple viewpoints rather than advocate for specific political positions.
    
    5. Transparency: Users should always be aware when interacting with politically biased AI systems.
    
    Consider using these techniques instead for:
    - Research into AI alignment and bias
    - Understanding how political biases emerge in language models
    - Developing better methods for detecting and mitigating bias
    - Creating educational demonstrations of AI bias
    
    Always ensure your AI development serves the broader public good.
    """
    main()
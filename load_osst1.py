import json
import pandas as pd
from datasets import Dataset, load_dataset
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

def _extract_conversation_path(tree_df: pd.DataFrame) -> Optional[List[Dict]]:
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
                'role': msg_data['role'] if msg_data['role']=='assistant' else 'user', # to convert prompter -> user
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
        print(f"Error processing conversation tree: {e}")
        return None

def _build_conversation_trees(df: pd.DataFrame) -> List[List[Dict]]:
    """Build conversation trees from the dataset"""
    conversations = []
    
    # Group by message tree id
    for message_tree_id in df['message_tree_id'].unique():
        tree_df = df[df['message_tree_id'] == message_tree_id].copy()
        
        # Build the conversation tree
        conversation = _extract_conversation_path(tree_df)
        if conversation:
            conversations.append(conversation)
            
    return conversations

# def _format_conversation(conversation: List[Dict]) -> str:
#     """Format conversation for training"""
#     formatted_parts = []
    
#     for msg in conversation:
#         role = msg['role']
#         text = msg['text'].strip()
        
#         if role == 'prompter':
#             formatted_parts.append(f"<|im_start|>user\n{text}<|im_end|>")
#         elif role == 'assistant':
#             formatted_parts.append(f"<|im_start|>assistant\n{text}<|im_end|>")
            
#     return "\n".join(formatted_parts)


def load_conversations(dataset_name):
    # dataset_name = 'OpenAssistant/oasst1'

    print(f"Loading dataset: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name, split="train").filter(lambda example: example['lang']=='en')

    # Convert to pandas for easier processing
    df = dataset.to_pandas()


    conversations = _build_conversation_trees(df)

    print(f"Processed {len(conversations)} conversations")
    print(json.dumps(conversations[0], indent=4))
    return conversations
    # Format conversations for training
    # formatted_conversations = [
    #     _format_conversation(conv) for conv in conversations
    # ]

    # print(formatted_conversations)

    # Filter out conversations that are too long
    # filtered_conversations = [
    #     conv for conv in formatted_conversations 
    #     if len(self.tokenizer.encode(conv)) <= self.max_seq_length
    # ]

    # print(f"Filtered to {len(filtered_conversations)} conversations within length limit")



if __name__=='__main__':
    load_conversations('OpenAssistant/oasst1')
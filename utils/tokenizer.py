from transformers import AutoTokenizer
import torch
import numpy as np

class Tokenizer:
    """
    Utility class for handling tokenization and detokenization.
    Wraps around Hugging Face's tokenizers with additional functionality.
    """
    
    def __init__(
        self,
        pretrained_model_name="bert-base-uncased",
        max_length=64,
        padding="max_length",
        truncation=True,
    ):
        """
        Initialize the tokenizer.
        
        Args:
            pretrained_model_name (str): Name of the pretrained tokenizer
            max_length (int): Maximum sequence length
            padding (str or bool): Padding strategy
            truncation (bool): Whether to truncate sequences
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
    def encode(self, text, return_tensors="pt"):
        """
        Encode text to token IDs.
        
        Args:
            text (str or list): Text or list of texts to encode
            return_tensors (str): Return format ('pt' for PyTorch, 'np' for NumPy, 'list' for lists)
            
        Returns:
            dict: Encoding including input_ids and attention_mask
        """
        if isinstance(text, str):
            text = [text]
            
        encoding = self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=return_tensors if return_tensors != "list" else None
        )
        
        return encoding
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens (bool): Whether to skip special tokens
            
        Returns:
            str or list: Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
            
        if token_ids.ndim == 1:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) 
                   for ids in token_ids]
    
    def batch_encode_plus(self, texts, **kwargs):
        """
        Encode a batch of texts.
        
        Args:
            texts (list): List of texts to encode
            **kwargs: Additional arguments for tokenization
            
        Returns:
            dict: Batch encoding
        """
        return self.tokenizer.batch_encode_plus(
            texts,
            padding=kwargs.get("padding", self.padding),
            truncation=kwargs.get("truncation", self.truncation),
            max_length=kwargs.get("max_length", self.max_length),
            return_tensors=kwargs.get("return_tensors", "pt")
        )
    
    def convert_tokens_to_ids(self, tokens):
        """
        Convert tokens to IDs.
        
        Args:
            tokens (str or list): Token or list of tokens
            
        Returns:
            int or list: Token IDs
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids, skip_special_tokens=True):
        """
        Convert IDs to tokens.
        
        Args:
            ids: Token IDs
            skip_special_tokens (bool): Whether to skip special tokens
            
        Returns:
            str or list: Tokens
        """
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self):
        """Get the vocabulary size."""
        return self.tokenizer.vocab_size
    
    @property
    def vocab(self):
        """Get the vocabulary."""
        return self.tokenizer.get_vocab()
    
    def save_pretrained(self, output_dir):
        """
        Save the tokenizer to a directory.
        
        Args:
            output_dir (str): Directory to save the tokenizer
        """
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a tokenizer from a pre-trained model or directory.
        
        Args:
            pretrained_model_name_or_path (str): Path or name of the pre-trained tokenizer
            **kwargs: Additional arguments
            
        Returns:
            Tokenizer: Loaded tokenizer
        """
        instance = cls(pretrained_model_name_or_path, **kwargs)
        return instance 
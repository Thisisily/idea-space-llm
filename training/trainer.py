import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import wandb
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

class Trainer:
    """
    Trainer for the Idea Space LLM model.
    Handles training, evaluation, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_epochs=10,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        log_steps=10,
        eval_steps=100,
        checkpoint_steps=1000,
        output_dir="./checkpoints",
        device=None,
        use_wandb=False,
        use_tensorboard=True,
        project_name="idea-space-llm",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The Idea Space LLM model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            warmup_ratio: Ratio of steps for learning rate warmup
            gradient_accumulation_steps: Number of steps for gradient accumulation
            log_steps: Steps between logging
            eval_steps: Steps between evaluation
            checkpoint_steps: Steps between saving checkpoints
            output_dir: Directory to save checkpoints
            device: Device to train on ('cuda', 'cpu', etc.) - auto-detected if None
            use_wandb: Whether to use Weights & Biases for logging
            use_tensorboard: Whether to use TensorBoard for logging
            project_name: Project name for logging
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.checkpoint_steps = checkpoint_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.project_name = project_name
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Setup dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            self.eval_dataloader = None
            
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Set up logging
        self._setup_logging()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Global step counter
        self.global_step = 0
        
    def _setup_optimizer(self):
        """Set up the AdamW optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
    
    def _setup_scheduler(self):
        """Set up learning rate scheduler with warmup."""
        num_training_steps = len(self.train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        return get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _setup_logging(self):
        """Set up logging with TensorBoard and/or W&B."""
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=f"{self.output_dir}/logs")
        else:
            self.tb_writer = None
            
        if self.use_wandb:
            wandb.init(project=self.project_name)
            wandb.watch(self.model)
    
    def _log_metrics(self, metrics, step):
        """Log metrics to TensorBoard and/or W&B."""
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
                    
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        # Also print metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        print(f"Step {step}: {metrics_str}")
    
    def save_checkpoint(self, step=None):
        """Save model checkpoint."""
        if step is None:
            step = self.global_step
            
        checkpoint_dir = f"{self.output_dir}/checkpoint-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), f"{checkpoint_dir}/model.pt")
        
        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), f"{checkpoint_dir}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{checkpoint_dir}/scheduler.pt")
        
        print(f"Saved checkpoint at step {step} to {checkpoint_dir}")
        
        if self.use_wandb:
            wandb.save(f"{checkpoint_dir}/*")
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return None
            
        self.model.eval()
        eval_loss = 0
        eval_rec_loss = 0
        eval_kl_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask")
                )
                
                eval_loss += outputs["loss"].item()
                eval_rec_loss += outputs["rec_loss"].item()
                if outputs["kl_loss"] is not None:
                    eval_kl_loss += outputs["kl_loss"].item()
        
        # Calculate average loss
        eval_loss /= len(self.eval_dataloader)
        eval_rec_loss /= len(self.eval_dataloader)
        eval_kl_loss /= len(self.eval_dataloader)
        
        metrics = {
            "eval_loss": eval_loss,
            "eval_rec_loss": eval_rec_loss,
            "eval_kl_loss": eval_kl_loss
        }
        
        self.model.train()
        return metrics
    
    def train(self):
        """Train the model."""
        self.model.to(self.device)
        self.model.train()
        
        progress_bar = tqdm(range(len(self.train_dataloader) * self.num_epochs))
        
        # Track losses for logging
        running_loss = 0.0
        running_rec_loss = 0.0
        running_kl_loss = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask")
                )
                
                loss = outputs["loss"]
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update running losses
                running_loss += loss.item() * self.gradient_accumulation_steps
                running_rec_loss += outputs["rec_loss"].item()
                if outputs["kl_loss"] is not None:
                    running_kl_loss += outputs["kl_loss"].item()
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    progress_bar.update(1)
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.log_steps == 0:
                        metrics = {
                            "loss": running_loss / self.log_steps,
                            "rec_loss": running_rec_loss / self.log_steps,
                            "kl_loss": running_kl_loss / self.log_steps if outputs["kl_loss"] is not None else 0,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "epoch": epoch + step / len(self.train_dataloader)
                        }
                        
                        self._log_metrics(metrics, self.global_step)
                        
                        running_loss = 0.0
                        running_rec_loss = 0.0
                        running_kl_loss = 0.0
                    
                    # Evaluation
                    if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, self.global_step)
                        
                        # Generate a sample text
                        self._log_sample_generation()
                    
                    # Checkpoint
                    if self.global_step % self.checkpoint_steps == 0:
                        self.save_checkpoint()
            
            # Evaluate at end of epoch
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, self.global_step)
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(step=f"epoch-{epoch + 1}")
            
        # Final evaluation and checkpoint
        if self.eval_dataloader is not None:
            eval_metrics = self.evaluate()
            self._log_metrics(eval_metrics, self.global_step)
            
        self.save_checkpoint(step="final")
        
        # Close loggers
        if self.tb_writer is not None:
            self.tb_writer.close()
            
        if self.use_wandb:
            wandb.finish()
            
        return self.global_step
    
    def _log_sample_generation(self, num_samples=2):
        """Generate and log sample text from the model."""
        self.model.eval()
        
        samples = []
        
        # Generate from random latent vectors
        for i in range(num_samples):
            z = torch.randn(1, self.model.latent_dim, device=self.device)
            
            with torch.no_grad():
                generated_text = self.model.generate(
                    z=z,
                    temperature=0.7,
                    top_k=50
                )
                
            samples.append({
                "random_z": generated_text
            })
            
        # Generate from encoded texts if we have eval data
        if self.eval_dataloader is not None:
            # Get a batch from eval
            eval_batch = next(iter(self.eval_dataloader))
            eval_batch = {k: v.to(self.device) for k, v in eval_batch.items()}
            
            # Encode a couple of examples
            with torch.no_grad():
                for i in range(min(num_samples, eval_batch["input_ids"].size(0))):
                    # Get the input text
                    input_ids = eval_batch["input_ids"][i:i+1]
                    attention_mask = eval_batch["attention_mask"][i:i+1] if "attention_mask" in eval_batch else None
                    
                    # Decode to get original text
                    input_text = self.model.tokenizer.decode(
                        input_ids[0], skip_special_tokens=True
                    )
                    
                    # Encode to latent space
                    z = self.model.encode(input_ids, attention_mask)
                    
                    # Generate from the latent
                    generated_text = self.model.generate(
                        z=z,
                        temperature=0.7,
                        top_k=50
                    )
                    
                    samples[i]["input_text"] = input_text
                    samples[i]["encoded_text"] = generated_text
        
        # Log samples
        if self.use_wandb:
            wandb.log({"samples": wandb.Table(
                columns=["random_z", "input_text", "encoded_text"],
                data=[[s.get("random_z", ""), s.get("input_text", ""), s.get("encoded_text", "")] for s in samples]
            )}, step=self.global_step)
        
        # Print samples
        print("\n=== Generated Samples ===")
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}:")
            if "input_text" in sample:
                print(f"  Input: {sample['input_text']}")
                print(f"  Encoded: {sample['encoded_text']}")
            print(f"  Random z: {sample['random_z']}")
            print()
            
        self.model.train() 
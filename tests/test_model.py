import torch
import unittest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Encoder, DiffusionDecoder, NoiseProcess, IdeaSpaceLLM
from utils.tokenizer import Tokenizer
from utils.embedding import EmbeddingUtils
from utils.latent_operations import LatentOperations

class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder(
            pretrained_model_name="prajjwal1/bert-tiny",  # Small model for testing
            latent_dim=64,
            pooling_strategy="mean",
            use_variational=True
        )
        
        self.tokenizer = Tokenizer(
            pretrained_model_name="prajjwal1/bert-tiny",
            max_length=16
        )
        
        self.test_texts = [
            "This is a test.",
            "Another test sentence."
        ]
    
    def test_initialization(self):
        self.assertEqual(self.encoder.latent_dim, 64)
        self.assertEqual(self.encoder.pooling_strategy, "mean")
        self.assertTrue(self.encoder.use_variational)
    
    def test_forward(self):
        encodings = self.tokenizer.encode(self.test_texts)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # Test forward pass
        z, kl_loss = self.encoder(input_ids, attention_mask)
        
        # Check shapes
        self.assertEqual(z.shape, (2, 64))  # Batch of 2, latent dim 64
        self.assertIsNotNone(kl_loss)
        
    def test_encode(self):
        encodings = self.tokenizer.encode(self.test_texts)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # Test encode method (for inference)
        z = self.encoder.encode(input_ids, attention_mask)
        
        # Check shape
        self.assertEqual(z.shape, (2, 64))


class TestNoiseProcess(unittest.TestCase):
    def setUp(self):
        self.noise_process = NoiseProcess(
            max_timesteps=100,
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=0.02
        )
        
        # Create some dummy embeddings
        self.batch_size = 2
        self.seq_len = 16
        self.embed_dim = 32
        self.x_0 = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
    def test_initialization(self):
        self.assertEqual(self.noise_process.max_timesteps, 100)
        self.assertEqual(self.noise_process.beta_schedule, "linear")
        self.assertEqual(len(self.noise_process.betas), 100)
    
    def test_q_sample(self):
        # Test forward diffusion
        t = torch.tensor([10, 50])
        x_t = self.noise_process.q_sample(self.x_0, t)
        
        # Check shape
        self.assertEqual(x_t.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # Check that x_t is different from x_0
        self.assertFalse(torch.allclose(x_t, self.x_0))


class TestDiffusionDecoder(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 64
        self.input_dim = 32
        self.hidden_dim = 128
        self.batch_size = 2
        self.seq_len = 16
        
        self.decoder = DiffusionDecoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2,  # Small for testing
            max_seq_len=self.seq_len
        )
        
        # Create dummy inputs
        self.x_t = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.t = torch.tensor([10, 50])
        self.z = torch.randn(self.batch_size, self.latent_dim)
        
    def test_initialization(self):
        self.assertEqual(self.decoder.input_dim, self.input_dim)
        self.assertEqual(self.decoder.hidden_dim, self.hidden_dim)
        self.assertEqual(self.decoder.latent_dim, self.latent_dim)
    
    def test_forward(self):
        # Test forward pass
        output = self.decoder(self.x_t, self.t, self.z)
        
        # Check shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))


class TestIdeaSpaceLLM(unittest.TestCase):
    def setUp(self):
        # Use a small model for faster testing
        self.model = IdeaSpaceLLM(
            pretrained_model_name="prajjwal1/bert-tiny",
            latent_dim=64,
            hidden_dim=128,
            embedding_dim=32,
            max_seq_len=16,
            diffusion_steps=10,  # Small for testing
            inference_steps=2,
            use_variational=True
        )
        
        self.test_texts = [
            "This is a test.",
            "Another test sentence."
        ]
        
    def test_initialization(self):
        self.assertEqual(self.model.latent_dim, 64)
        self.assertEqual(self.model.max_seq_len, 16)
        self.assertEqual(self.model.diffusion_steps, 10)
        
    def test_encode(self):
        # Tokenize
        encodings = self.model.tokenizer(
            self.test_texts,
            padding="max_length",
            truncation=True,
            max_length=self.model.max_seq_len,
            return_tensors="pt"
        )
        
        # Encode
        z = self.model.encode(encodings.input_ids, encodings.attention_mask)
        
        # Check shape
        self.assertEqual(z.shape, (2, 64))  # Batch of 2, latent dim 64
    
    def test_forward(self):
        # Tokenize
        encodings = self.model.tokenizer(
            self.test_texts,
            padding="max_length",
            truncation=True,
            max_length=self.model.max_seq_len,
            return_tensors="pt"
        )
        
        # Forward pass
        outputs = self.model(encodings.input_ids, encodings.attention_mask)
        
        # Check outputs
        self.assertIn("loss", outputs)
        self.assertIn("rec_loss", outputs)
        self.assertIn("kl_loss", outputs)
        self.assertIn("pred", outputs)
        self.assertIn("target", outputs)
        self.assertIn("z", outputs)
        
    @unittest.skip("Generator test is slow and requires extra memory")
    def test_generate(self):
        # Tokenize
        encodings = self.model.tokenizer(
            self.test_texts[0],  # Just use one text
            padding="max_length",
            truncation=True,
            max_length=self.model.max_seq_len,
            return_tensors="pt"
        )
        
        # Encode
        z = self.model.encode(encodings.input_ids, encodings.attention_mask)
        
        # Generate
        generated_text = self.model.generate(z=z, temperature=1.0)
        
        # Check that we got a string back
        self.assertIsInstance(generated_text, str)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.dim = 64
        self.a = torch.randn(self.batch_size, self.dim)
        self.b = torch.randn(self.batch_size, self.dim)
        
    def test_embedding_utils(self):
        # Test cosine similarity
        sim = EmbeddingUtils.cosine_similarity(self.a, self.b)
        self.assertEqual(sim.shape, (self.batch_size, self.batch_size))
        
        # Test interpolation
        interp = EmbeddingUtils.interpolate(self.a[0], self.b[0], alpha=0.5)
        self.assertEqual(interp.shape, (self.dim,))
        
        # Test dimensionality reduction
        reduced = EmbeddingUtils.reduce_dimensions(self.a.numpy(), method="pca", n_components=2)
        self.assertEqual(reduced.shape, (self.batch_size, 2))
        
    def test_latent_operations(self):
        # Test interpolation
        interp = LatentOperations.interpolate(self.a[0], self.b[0], alpha=0.5)
        self.assertEqual(interp.shape, (self.dim,))
        
        # Test batch interpolation
        interp_batch = LatentOperations.interpolate_batch(self.a[0], self.b[0], steps=5)
        self.assertEqual(interp_batch.shape, (5, self.dim))
        
        # Test arithmetic
        result = LatentOperations.arithmetic([self.a[0], self.b[0]], [1.0, -0.5])
        self.assertEqual(result.shape, (self.dim,))
        
        # Test noise addition
        noisy = LatentOperations.add_noise(self.a[0], scale=0.1)
        self.assertEqual(noisy.shape, (self.dim,))
        self.assertFalse(torch.allclose(noisy, self.a[0]))


if __name__ == "__main__":
    unittest.main() 
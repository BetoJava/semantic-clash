"""
Setup script to download and prepare the Qwen3-Embedding-0.6B model.
This script downloads the model so it's ready for use.
"""

import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch


def setup_model(model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cpu"):
    """
    Download and prepare the embedding model.
    
    Args:
        model_name: Name of the HuggingFace model to download
        device: Device to use ('cpu' or 'cuda')
    """
    print("=" * 60)
    print("Qwen3 Embedding Model Setup")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print()
    
    # Ensure CPU is used if specified
    if device == "cpu":
        torch.set_num_threads(1)
        print("CPU mode enabled")
    
    print(f"\nDownloading model from HuggingFace...")
    print("This may take a few minutes depending on your internet connection.")
    print()
    
    try:
        # Load the model - this will download it if not already cached
        model = SentenceTransformer(model_name, device=device)
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"  Model dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test the model with a simple example
        print(f"\nTesting model with sample text...")
        test_texts = ["This is a test sentence.", "This is another test sentence."]
        embeddings = model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True)
        
        print(f"✓ Model test successful!")
        print(f"  Test embeddings shape: {embeddings.shape}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        print("\n" + "=" * 60)
        print("Setup complete! The model is ready to use.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during model setup: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI entry point for setup script."""
    parser = argparse.ArgumentParser(
        description="Setup script to download and prepare Qwen3-Embedding-0.6B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download model for CPU usage (default)
  python setup.py
  
  # Download model for GPU usage
  python setup.py --device cuda
  
  # Download a different model
  python setup.py --model Qwen/Qwen3-Embedding-4B
        """
    )
    
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model name (default: Qwen/Qwen3-Embedding-0.6B)"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)"
    )
    
    args = parser.parse_args()
    
    success = setup_model(args.model, args.device)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()


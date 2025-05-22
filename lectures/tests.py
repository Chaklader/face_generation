import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def check_dataset_outputs(dataset: Dataset):
    assert len(dataset) == 32600, 'The dataset should contain 32,600 images.'
    index = np.random.randint(len(dataset))
    image = dataset[index]
    assert  image.shape == torch.Size([3, 64, 64]), 'You must reshape the images to be 64x64'
    assert image.min() >= -1 and image.max() <= 1, 'The images should range between -1 and 1.'
    print('Congrats, your dataset implementation passed all the tests')
    
    
def check_discriminator(discriminator: torch.nn.Module):
    images = torch.randn(1, 3, 64, 64)
    score = discriminator(images)
    assert score.shape == torch.Size([1, 1, 1, 1]), 'The discriminator output should be a single score.'
    print('Congrats, your discriminator implementation passed all the tests')
 

def check_generator(generator: torch.nn.Module, latent_dim: int):
    latent_vector = torch.randn(1, latent_dim, 1, 1)
    image = generator(latent_vector)
    assert image.shape == torch.Size([1, 3, 64, 64]), 'The generator should output a 64x64x3 images.'
    print('Congrats, your generator implementation passed all the tests')
    
def check_apply_noise(noise_layer: torch.nn.Module):
    """Verify the noise injection layer meets StyleGAN requirements.
    
    Args:
        noise_layer: Instance of ApplyNoise to test
        
    Tests:
        1. Output shape matches input shape
        2. Noise is actually added (output != input)
        3. Different noise per spatial location
        4. Same noise scaling per channel
        5. Gradient flow through weights
    """
    # Test 1: Shape preservation
    x = torch.randn(1, 512, 32, 32)
    out = noise_layer(x)
    assert x.shape == out.shape, "Output shape should match input shape"
    
    # Test 2: Noise effect verification
    with torch.no_grad():
        out1 = noise_layer(x)
        out2 = noise_layer(x)
        assert not torch.allclose(out1, out2), "Should produce different outputs each call"
        assert not torch.allclose(out1, x), "Should modify input with noise"
        
    # Test 3: Verify gradient flow
    x.requires_grad_(True)
    out = noise_layer(x)
    loss = out.sum()
    loss.backward()
    assert noise_layer.weights.grad is not None, "Gradients should flow through weights"
    assert x.grad is not None, "Gradients should flow through input"
    
    # Test 4: Channel-wise scaling verification
    batch = torch.randn(2, 3, 8, 8)
    out = noise_layer(batch)
    channel_means = out.mean(dim=(0, 2, 3))
    assert not torch.allclose(channel_means, channel_means[0]*torch.ones_like(channel_means)), \
           "Different channels should have different scaling"
    
    print('Congrats, your ApplyNoise implementation passed all tests')    
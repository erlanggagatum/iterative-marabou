from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from maraboupy import Marabou

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import onnx

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import math
import json

label_dict = {
    0:'T-shirt/top',
    1:'Trouser',
    2:'Pullover',
    3:'Dress',
    4:'Coat',
    5:'Sandal',
    6:'Shirt',
    7:'Sneaker',
    8:'Bag',
    9:'Ankle boot'
}

model_file = 'simple_nn_fashion_mnist_sequential_50e'
# model_file = 'simple_nn_fashion_mnist_sequential'

def load_model():
    model = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(28*28, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    
    model.load_state_dict(torch.load(f'{model_file}.pth', weights_only=True))
    model.eval()
    return model

def verify_model(image, label, network, epsilon, beta, max_epsilon=0.5, max_iteration=10):
    # This code contain the implementation of
    # Iterative marabou verification
    logs = []
    for _ in range(max_iteration):
        if epsilon >= max_epsilon:
            logs.append({
                'epsilon': max_epsilon,
                'beta': beta,
                'max_iteration': max_iteration,
                'max_epsilon': max_epsilon,
                'result': '',
                'value': '',
                # 'statistics': None
            })

            print('UNSAT: Reached maximum epsilon')
            break
        # bounds generation
        def generate_bounds(image, epsilon=0.1):
            # 1 Dim image (black and white)
            if (image.shape[0] == 1):
                image = image[0]

            # Set lower and upper bound
            lowerBounds = image-epsilon
            upperBounds = image+epsilon
            
            # Clip the bound to match the minimum and maximum value from normalization
            # minimum: -1, and maximum 1
            lowerBounds = np.clip(lowerBounds, -1, 1)
            upperBounds = np.clip(upperBounds, -1, 1)

            return lowerBounds.flatten(), upperBounds.flatten()

        # generate lower bound
        lowerBounds, upperBounds = generate_bounds(image, epsilon)

        input_vars = network.inputVars[0].flatten()
        for input_var_index in input_vars:
            network.setLowerBound(input_var_index, lowerBounds[input_var_index])
            network.setUpperBound(input_var_index, upperBounds[input_var_index])

        # Get output variable IDs
        output_var = network.outputVars[0][0]
        
        # Add a constraint that output[0] > output[i] for all i != 0
        for i in range(10):
            if i != label:  # Class label is the expected class
                network.addInequality([output_var[label], output_var[i]], [1, -1], 0)

        # create options
        opt = Marabou.createOptions(timeoutInSeconds=180)
        
        # Find the solution using network.solve()
        result, values, statistics = network.solve(verbose=False, options=opt)

        logs.append({
            'epsilon': epsilon,
            'beta': beta,
            'max_iteration': max_iteration,
            'max_epsilon': max_epsilon,
            'result': result if result in ['sat','unsat'] else '',
            'value': values if result in ['sat','unsat'] else '',
            # 'statistics': statistics if result in ['sat','unsat'] else None
        })

        # Interpret the results
        if result == "sat":
            print("SAT: A solution was found that meets the constraints.")
            print("Variable values satisfying the constraints:")
            break
        else:
            print("UNSAT: No solution satisfies the constraints.")
        epsilon = round(epsilon + beta, 2)
    
    return logs

def visualize_image(original_input, perturbed_input, original_label, perturbed_label):
    
    fig = plt.figure(figsize=(8, 4))
    
    fig.add_subplot(1, 2, 1)
    plt.title(f'Original ({label_dict[original_label]})')
    plt.imshow(original_input)
    
    fig.add_subplot(1, 2, 2)
    plt.title(f'Perturbed ({label_dict[perturbed_label]})')
    plt.imshow(perturbed_input)
    
    plt.savefig('perturbed-img.png', dpi=300, bbox_inches='tight')
    
def visualize_embeddings(model, real_class_img, counter_class_img, original_input, perturbed_input):
    
    # Extract embeddings for original and perturbed input (an image with its perturbed version)
    original_input_emb = model[:-1](torch.Tensor(original_input[np.newaxis, ...])).detach().numpy()
    perturbed_input_emb = model[:-1](torch.Tensor(perturbed_input[np.newaxis, ...])).detach().numpy()
    
    # Extract embeddings for original and perturbed inputs (images from test set)
    real_class_emb = model[:-1](torch.Tensor(np.array(real_class_img))).detach().numpy()
    counter_class_emb = model[:-1](torch.Tensor(np.array(counter_class_img))).detach().numpy()


    # Combine embeddings
    embs = np.concatenate((real_class_emb, counter_class_emb, original_input_emb, perturbed_input_emb))

    # Create new labels
    real_class_label = 0
    counter_class_label = 1
    original_input_label = 2
    perturbed_input_label = 3

    labels = np.concatenate((
        np.full((len(real_class_emb),), real_class_label),         # Label 0: Real Class
        np.full((len(counter_class_emb),), counter_class_label),   # Label 1: Counter Class
        np.full((1,), original_input_label),                       # Label 2: Original Input
        np.full((1,), perturbed_input_label)                       # Label 3: Perturbed Input
    ))

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embs)

    # Create custom colors for each group
    colors = ['#c7e0ed', '#edc7c7', 'green', 'red']
    scatter_colors = [colors[int(label)] for label in labels]

    # Visualize the result
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=scatter_colors, alpha=1
    )

    # Create custom legend
    legend_labels = [
        'Real Class Images (Blue)',
        'Counter Class Images (Pink)',
        'Original Input (Green)',
        'Perturbed Input (Red)'
    ]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    plt.legend(handles, legend_labels, loc="lower left")

    # Add title and display plot
    plt.title("t-SNE Visualization of Fashion-MNIST Embeddings (Real image vs perturbed image)")
    plt.savefig('perturbed-embedding.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    
    # This method contains the full implementation of:
    #    1. Iterative marabou evaluation
    #    2. Input image visualization
    #    3. Embedding visualization
    
    data = []
    beta = 0.02 
    epsilon = 0.01
    max_iteration = 20
    max_epsilon = 0.5
    filename = 'verification-logs-2'

    # Load pretrained model weights
    model = load_model()
    
    # Load image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load Fashion-MNIST dataset
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    # DataLoader
    batch_size = 64
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print('Start verification')
    print('Epsilon:', epsilon)
    print('Beta:', beta)
    
    stop_idx = 0
    
    # Run data on 1 batch only (which contains 64 data)
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            if stop_idx == 50: # after this run 50
                break
            stop_idx+=1
            
            # get image and its label
            image = image.numpy()
            label = label.item()

            # Load the exported ONNX model in Marabou
            onnx_file_path = f"{model_file}.onnx"
            network = Marabou.read_onnx(onnx_file_path)
            
            # verify to n number of iteration
            logs = verify_model(image, label, network, epsilon, beta, max_epsilon=max_epsilon, max_iteration=max_iteration)
            
            temp_data = {
                'verification_logs': logs,
                'original_label': '',
                'perturbed_label': ''
            }
            
            # Visualization (Can be performed only if the result is SAT)
            if logs[-1]['result'] == 'sat':
                
                # ==== Preprocessing before visualization ====
                # Get original and perturbed image based on log data
                perturbed_input = np.array(list(logs[-1]['value'].values()))[:784] # Image dimension is 28 x 28, so take the first 784 data
                perturbed_input = perturbed_input.reshape(28, 28) # Reshape X' from (784,) to (28,28)
                original_input = image[0]
                
                # Get counter class by predicting the perturbed input class
                counter_class = model(torch.Tensor(perturbed_input[np.newaxis, ...])).detach().numpy()[0]
                counter_class = np.argmax(counter_class)
                
                # Save all test image
                all_test_images = []
                all_test_labels = []
                for test_images, test_labels in test_loader:
                    all_test_labels.extend(test_labels.tolist())
                    all_test_images.extend(test_images.tolist())
                all_test_images = np.array(all_test_images)

                real_class_img = []
                real_labels = []
                counter_class_img = []
                counter_labels = []

                # Pick the real class image with its label, and its counterexample.
                for _, (i, l) in enumerate(zip(all_test_images, all_test_labels)):
                    # Get real class image and label
                    if l == label:
                        real_class_img.append(i)
                        real_labels.append(l)
                        continue
                    # Get counter class and label
                    elif l == counter_class:                         
                        counter_class_img.append(i)
                        counter_labels.append(l)
                        continue
                
                # Visualize the image
                visualize_image(original_input, perturbed_input, real_labels[0], counter_labels[0])
                
                # Visualize the embedding
                visualize_embeddings(model, real_class_img, counter_class_img, original_input, perturbed_input)
                
                temp_data['original_label'] = label
                temp_data['perturbed_label'] = counter_class
            
            # Append temp_data to data to save all verification result
            data.append(temp_data)
        break
    
    # Save the result to verification-logs.json
    with open(f'{filename}.json', 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)

    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert NumPy types to native Python types
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        return super(NumpyEncoder, self).default(obj)
    
if __name__ == '__main__':
    main()

# Marabou Implementation

This is a submission by Gatum Erlangga for Neural Network Verification class

## Requirements
The code is run on ```python==3.10```, with ```conda``` environment. To create ```conda``` environment, run the following command:
```
conda create -n marabou python==3.10
```
After that, install the following libraries.
```
numpy==1.26.4
onnx==1.17.0
onnxruntime==1.19.2
tf2onnx==1.16.1
scikit-learn==1.5.2
maraboupy==2.0.0 (latest)
```
This project uses the latest PyTorch stable version at the submission (2.5.1), which is  installed using the following command:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


## Code Implementations
- `main.py`: Contains the full implementation of: 1. Iterative verification `main()`; 2. Input visualization `visualize_image()`; and 3. Embedding visualization `visualize_embeddings()`
- `fp_nn_fashion_mnist_model_training.ipynb`: Model and training code
- `analysis.ipynb`: Analysis file to produce 4 figures on the report (Result and discussion)

## Run
Run main.py from its parent directory using the following command:
```
python main.py
```

## Author
- [@erlanggagatum](https://www.github.com/erlanggagatum)
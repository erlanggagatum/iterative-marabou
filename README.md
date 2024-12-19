
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


## Files and Directories
- `/data`: Contains dataset location. This directory will be filled after running the `main.py` script, which contains a code to download the dataset, and put it inside the directory.
- `*.png`: Figures resulted from analysis and verification.
- `*.onnx`: Exported model in `ONNX` format required for marabou verification.
- `*.pth` and `*.pt`: Exported model in PyTorch format. There are two models that can be used, such as `simple_nn_fashion_mnist_sequential` and `simple_nn_fashion_mnist_sequential_50e`. The only difference is the first model was trained for 10 epoch, meanwhile the other one is 50 epoch. To use one of the model, simply modify `model_file` variable with the desired model. The default model is the 50 epoch trained model.

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
# ML-model-loader
This is a ML weights and biases loader directly on CPU. It only works on the linear neural network and also contains INT8 quantisation model along with a trail python code to run.

1st let's see how to use the given files:
- Uncomment the ML_loader_3.cpp main file and add the weights "multi_model_weights.bin" in same folder as of the ml loader.
- The "quantized_weights.bin" can also be used after changing the file location in the main of ML_loader_3.cpp.
- ADD the JSON file into the VS Code so that it support the SIMD.
- Build the file.
- Run the .cpp by "./ML_loader_3" in the terminal.

Let's talk about how to run this example python code and see if it runs.
- Run the colab file.
- Then when you recive the multi_model_weights.bin, add it into the same folder as your .cpp.

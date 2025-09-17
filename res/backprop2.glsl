#version 450 core
#define INPUT_DATA_COUNT 10
#define LAYERS 4
#define NEURONS 3  // the count of neurons in the largest layer

layout (local_size_x = NEURONS, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer RunOutputs {
    float[] runOutputs;
};

layout(std430, binding = 1) buffer Labels {
    float[] labels;
};

layout(std430, binding = 2) buffer Weight {
    float[] weights;
};

layout(std430, binding = 3) buffer Bias {
    float[] biases;
};

uniform Layer[LAYERS] layers;

void backpropOutputLayer(uint neuronId) {
    int layer = LAYERS-1;
    int prevLayer = layer-1;
}

void main() {
    uint neuronId = gl_LocalInvocationID.x;

    backpropOutputLayer(neuronId);
}

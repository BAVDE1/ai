#version 450 core
#define LAYERS 4
#define NEURONS 3  // the count of neurons in the largest layer

layout (local_size_x = NEURONS, local_size_y = 1, local_size_z = 1) in;

struct Layer {
    int size;
    int weightsOffset;
    int biasesOffset;
};

layout(std430, binding = 0) buffer Input {
    float[] inputs;
};

layout(std430, binding = 1) buffer Weight {
    float[] weights;
};

layout(std430, binding = 2) buffer Bias {
    float[] biases;
};

layout(std430, binding = 3) buffer Output {
    float[] outputs;
};

uniform Layer[LAYERS] layers;

shared float[NEURONS] activationCache;

float sigmoid(float value) {
    return 1 / (1 + exp(-1 * value));
}

float nodeValue(uint weightPos, uint biasPos, int prevLayerSize) {
    float val = 0;
    for (uint n = 0; n < prevLayerSize; n++) {
        val += weights[weightPos + n] * activationCache[n];
    }
    val += biases[biasPos];
    return val;
}

void waitOnSync() {
    memoryBarrierShared();
    barrier();
}

void main() {
    int size = layers[0].size;
    uint neuronId = gl_LocalInvocationID.x;
    uint inputInx = gl_WorkGroupID.x * size;

    // enter input values ready for layer 1
    if (neuronId < size) {
        activationCache[neuronId] = inputs[inputInx + neuronId];
    }

    // feed forward!
    for (int l = 1; l < LAYERS; l++) {
        waitOnSync();
        size = layers[l - 1].size;  // the previous layer size
        uint weightPos = layers[l].weightsOffset + neuronId;
        uint biasPos = layers[l].biasesOffset + neuronId;
        float value = nodeValue(weightPos, biasPos, size);
        waitOnSync();  // cause were using activationCache to find the value
        activationCache[neuronId] = sigmoid(value);
    }

    // output the last layer's values
    size = layers[LAYERS - 1].size;
    if (neuronId < size) {
        outputs[inputInx + neuronId] = activationCache[neuronId];
    }
}

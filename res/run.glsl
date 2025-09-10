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
    for (uint prevNeuronId = 0; prevNeuronId < prevLayerSize; prevNeuronId++) {
        val += weights[weightPos + prevNeuronId] * activationCache[prevNeuronId];
    }
    val += biases[biasPos];
    return val;
}

void waitOnSync() {
    memoryBarrierShared();
    barrier();
}

void main() {
    uint neuronId = gl_LocalInvocationID.x;
    uint inputId = gl_WorkGroupID.x;
    int size = layers[0].size;

    // enter input values ready for layer 1
    if (neuronId < size) {
        activationCache[neuronId] = inputs[(inputId * size) + neuronId];
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

    // output the last layer's activated values
    size = layers[LAYERS - 1].size;
    if (neuronId < size) {
        outputs[(inputId * size) + neuronId] = activationCache[neuronId];
    }
}

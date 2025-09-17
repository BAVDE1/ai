#version 450 core
#define LAYERS 4
#define NEURONS 3  // the count of neurons in the largest layer

layout (local_size_x = NEURONS, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer Input {
    float[] inputs;
};

layout(std430, binding = 1) buffer Weight {
    float[] weights;
};

layout(std430, binding = 2) buffer Bias {
    float[] biases;
};

layout(std430, binding = 3) buffer OutputA {
    float[] outputs;
};

layout(std430, binding = 4) buffer OutputB {
    float[] activationHistoryOutput;
};

struct Layer {
    int size;
    int weightsOffset;
    int biasesOffset;
};

uniform Layer[LAYERS] layers;

shared float[NEURONS] activationCache;
// goddamit
shared float[LAYERS][NEURONS] activationHistoryCache;  // history of all activated values for each neuron in each layer

float sigmoid(float value) {
    return 1 / (1 + exp(-1 * value));
}

float neuronValue(uint weightPos, uint biasPos, int prevLayerSize) {
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
        size = layers[l - 1].size;
        uint weightPos = layers[l].weightsOffset + neuronId;
        uint biasPos = layers[l].biasesOffset + neuronId;
        float value = neuronValue(weightPos, biasPos, size);
        waitOnSync();
        activationCache[neuronId] = sigmoid(value);
    }

    // output the last layer's activated values
    size = layers[LAYERS - 1].size;
    if (neuronId < size) {
        outputs[(inputId * size) + neuronId] = activationCache[neuronId];
    }
}

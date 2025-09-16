#version 450 core
#define INPUT_DATA_COUNT 10
#define INPUT_DATA_COUNT_AVG 1 / INPUT_DATA_COUNT
#define LAYERS 4
#define NEURONS 3  // the count of neurons in the largest layer

layout (local_size_x = NEURONS, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer Input {
    float[] inputs;
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

layout(std430, binding = 4) buffer Output {
    float[] outputs;
};

struct Layer {
    int size;
    int weightsOffset;
    int biasesOffset;
};

struct BackpropValues {
    float weightGradient;
    float biasGradient;
    float propagator;
};

uniform Layer[LAYERS] layers;

shared float[LAYERS][NEURONS] activationHistoryCache;  // history of all activated values for each neuron in each layer
//shared float[LAYERS][NEURONS] activationHistoryCacheAvgs;
shared float[INPUT_DATA_COUNT] costCache;
shared BackpropValues[LAYERS][NEURONS] backpropValues;
shared float cost;

// binary cross entropy cost
float costBCE(float prediction, float actual) {
    return actual * log(prediction) + (1 - actual) * log(1 - prediction);
}

float sigmoid(float value) {
    return 1 / (1 + exp(-1 * value));
}

float neuronValue(uint weightPos, uint biasPos, int prevLayerSize, int layerNum) {
    float val = 0;
    for (uint prevNeuronId = 0; prevNeuronId < prevLayerSize; prevNeuronId++) {
        val += weights[weightPos + prevNeuronId] * activationHistoryCache[layerNum-1][prevNeuronId];
    }
    val += biases[biasPos];
    return val;
}

BackpropValues backpropOutputLayer(uint neuronId, uint inputId) {
    int layer = LAYERS-1;
    int prevLayer = layer-1;
    float activatedValue = activationHistoryCache[layer][neuronId];  // AL
    float dC_dZLi = INPUT_DATA_COUNT_AVG * (activatedValue - labels[inputId]);

    float dC_WLi = 0;
    for (int i = 0; i < NEURONS; i++) {
        if (i >= layers[prevLayer].size) break;
        dC_WLi += dC_dZLi * activationHistoryCache[prevLayer][i];
    }

    // only use one thread to do bias
    if (neuronId == 0) {

    }
    float dC_bLi = 0;

//    float dC_dWL = dC_dZL *
    // uuuuuh
    return BackpropValues(0, 0, 0);
}

BackpropValues backpropLayer(int layer, uint neuronId) {
    return BackpropValues(0, 0, 0);
}

// blocks until all threads have reached this call
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
        activationHistoryCache[0][neuronId] = inputs[(inputId * size) + neuronId];
    }

    // feed forward!
    for (int l = 1; l < LAYERS; l++) {
        waitOnSync();
        size = layers[l - 1].size;
        uint weightPos = layers[l].weightsOffset + neuronId;
        uint biasPos = layers[l].biasesOffset + neuronId;
        float value = neuronValue(weightPos, biasPos, size, l);
        waitOnSync();
        activationHistoryCache[l][neuronId] = sigmoid(value);
    }

    // compute costs
    waitOnSync();
    float prediction = activationHistoryCache[LAYERS-1][neuronId];
    float actual = labels[inputId];
    costCache[inputId] = costBCE(prediction, actual);
    waitOnSync();

    // use one thread to average costs
    if (inputId == 0) {
        float totalCosts = 0;
        for (int i = 0; i < INPUT_DATA_COUNT; i++) totalCosts += costCache[i];
        cost = totalCosts / INPUT_DATA_COUNT;
    }
    waitOnSync();

    // calc backpropagation values
    backpropValues[LAYERS-1][neuronId] = backpropOutputLayer(neuronId, inputId);
    for (int l = LAYERS-2; l > 0; l--) {
        size = layers[l].size;
        if (neuronId < size) {
            backpropValues[l][neuronId] = backpropLayer(l, neuronId);
        }
        waitOnSync();
    }

    // output backpropogation values for each layer
}

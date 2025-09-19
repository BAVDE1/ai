#version 450 core
#define INPUT_DATA_COUNT 10
#define INPUT_DATA_COUNT_AVG 1. / INPUT_DATA_COUNT
#define LAYERS 4
#define NEURONS 3  // the count of neurons in the largest layer

layout (local_size_x = NEURONS, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 4) buffer Input {
    float[] inputs;
};

layout(std430, binding = 5) buffer Labels {
    float[] labels;
};

layout(std430, binding = 6) buffer Weight {
    float[] weights;
};

layout(std430, binding = 7) buffer Bias {
    float[] biases;
};

layout(std430, binding = 8) buffer WeightOutput {
    float[] weightOutputs;
};

layout(std430, binding = 9) buffer BiasOutput {
    float[] biasOutputs;
};

layout(std430, binding = 10) buffer Output {
    float[] outputs;
};

//layout(std430, binding = 6) buffer PropagatorOutput {
//    float[] propagatorOutputs;
//};

struct Layer {
    int size;
    int weightsOffset;
    int biasesOffset;
};

//struct BackpropValues {
//    float[NEURONS] weightGradient;
//    float biasGradient;
//    float propagator;
//};

uniform Layer[LAYERS] layers;

shared float[LAYERS][NEURONS] activationHistoryCache;  // history of all activated values for each neuron in each layer
//shared float[LAYERS][NEURONS] activationHistoryCacheAvgs;
//shared BackpropValues[LAYERS][NEURONS] backpropValues;

shared float[LAYERS][NEURONS] weightGradients;
shared float[LAYERS][NEURONS] biasGradients;
shared float[NEURONS][NEURONS] propagatorsCache;
shared float[NEURONS] propagators;

//shared float[NEURONS] costCache;
//shared float cost;  // this inputs' cost

// binary cross entropy cost
//float costBCE(float prediction, float actual) {
//    return actual * log(prediction) + (1 - actual) * log(1 - prediction);
//}

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

// blocks until all threads have reached this call
void waitOnSync() {
    memoryBarrierShared();
    barrier();
}

void backpropOutputLayer(uint neuronId, uint inputId) {
    int layerInx = LAYERS-1;
    int prevLayerInx = layerInx-1;
    Layer layer = layers[layerInx];
    Layer prevLayer = layers[prevLayerInx];
    float activatedValue = activationHistoryCache[layerInx][neuronId];  // AL
    float dC_dZLi = INPUT_DATA_COUNT_AVG * (activatedValue - labels[inputId]);  // error

//    float[] dC_WLi;
    for (int i = 0; i < prevLayer.size; i++) {
        weightGradients[layerInx][neuronId] += dC_dZLi * activationHistoryCache[prevLayerInx][i];
    }

    // bias is done in learn
    biasGradients[layerInx][neuronId] = dC_dZLi;

//    float propagator = 0;
    for (int i = 0; i < prevLayer.size; i++) {
        propagatorsCache[neuronId][i] = weights[layer.weightsOffset + i] * dC_dZLi;
    }
    waitOnSync();

    // add together atomically
    if (neuronId == 0) {
        for (int li = 0; li < layer.size; li++) {
            for (int i = 0; i < prevLayer.size; i++) {
                propagators[i] += propagatorsCache[li][i];
            }
        }
    }
    waitOnSync();
//    propagators[neuronId] = propagator;
//    if (neuronId == 0) {
//        for (int l = 0; l < layers[layer].size; l++) {
//            biasGradients[layer][neuronId] += dC_dZLi;
//        }
//    }
//    float dC_bLi = 0;

//    float dC_dWL = dC_dZL *
    // uuuuuh
//    return BackpropValues(dC_WLi, 0, 0);
}

void backpropLayer(int layerInx, uint neuronId, uint inputId) {
    int prevLayerInx = layerInx-1;
    Layer layer = layers[layerInx];
    Layer prevLayer = layers[prevLayerInx];
    float activatedValue = activationHistoryCache[layerInx][neuronId];  // Al

    float propagator = propagators[neuronId];  // dC_dZ_p1
    float dC_dZli = propagator * (activatedValue * (1 - activatedValue));  // using sigmoid derivative

    for (int i = 0; i < prevLayer.size; i++) {
        weightGradients[layerInx][neuronId] += dC_dZli * activationHistoryCache[prevLayerInx][i];
    }

    biasGradients[layerInx][neuronId] = dC_dZli;

    waitOnSync();  // wait before assinging new propagators
    for (int i = 0; i < prevLayer.size; i++) {
        propagatorsCache[neuronId][i] = weights[layer.weightsOffset + i] * dC_dZli;
    }
    waitOnSync();

    // add together atomically
    if (neuronId == 0) {
        for (int li = 0; li < layer.size; li++) {
            for (int i = 0; i < prevLayer.size; i++) {
                propagators[i] += propagatorsCache[li][i];
            }
        }
    }
    waitOnSync();
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

//    size = layers[LAYERS - 1].size;
//    if (neuronId < size) {
//        outputs[(inputId * size) + neuronId] = activationHistoryCache[3][neuronId];
//    }

    // compute costs
//    waitOnSync();
//    float prediction = activationHistoryCache[LAYERS-1][neuronId];
//    float actual = labels[inputId];
//    costCache[neuronId] = costBCE(prediction, actual);
//    waitOnSync();
//
//    // use one thread to average costs of this input
//    if (neuronId == 0) {
//        float totalCosts = 0;
//        for (int i = 0; i < NEURONS; i++) totalCosts += costCache[i];
//        cost = totalCosts / layers[LAYERS-1].size;
//    }
    waitOnSync();

    // calc backpropagation values
    backpropOutputLayer(neuronId, inputId);
    waitOnSync();
    backpropLayer(2, neuronId, inputId);
    waitOnSync();
    backpropLayer(1, neuronId, inputId);
    waitOnSync();

    // flatten weights and biases for output
    uint inputInx = inputId * (LAYERS * NEURONS);
    for (int l = 0; l < LAYERS; l++) {
        uint layerInx = l * NEURONS;
//        for (int n = 0; n < NEURONS; n++) {
        weightOutputs[inputInx + layerInx + neuronId] = weightGradients[l][neuronId];
        biasOutputs[inputInx + layerInx + neuronId] = biasGradients[l][neuronId];
//        }
    }
}

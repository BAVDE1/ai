#version 450 core
#define INPUT_DATA_COUNT 10
#define INPUT_DATA_COUNT_AVG 1. / INPUT_DATA_COUNT
#define LAYERS 4
#define NEURONS 3  // the count of neurons in the largest layer

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 10) buffer WeightInput {
    float[] weightInputs;
};

layout(std430, binding = 11) buffer BiasInput {
    float[] biasInputs;
};

layout(std430, binding = 12) buffer WeightOutput {
    float[] weightOutput;
};

layout(std430, binding = 13) buffer BiasOutput {
    float[] biaseOutput;
};

struct Layer {
    int size;
    int weightsOffset;
    int biasesOffset;
};

uniform Layer[LAYERS] layers;

void main() {

}

import org.ejml.equation.Equation;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;

public class MainCPU {
    static public class BackPropValues {
        SimpleMatrix weightGradient;
        SimpleMatrix biasGradient;
        SimpleMatrix propagator;

        public BackPropValues(SimpleMatrix weightGradient, SimpleMatrix biasGradient, SimpleMatrix propagator) {
            this.weightGradient = weightGradient;
            this.biasGradient = biasGradient;
            this.propagator = propagator;
        }
    }

    static final int EPOCHS = 1000;
    static final double ALPHA = 0.1;  // learning rate
    static final ArrayList<Double> costHistory = new ArrayList<>();
    static final HashMap<Integer, SimpleMatrix> activationCache = new HashMap<>();

    static int trainingDataCount = 10;
    static int[] layerNodes = new int[]{2, 3, 3, 1};

    static SimpleMatrix[] weights = new SimpleMatrix[layerNodes.length];
    static SimpleMatrix[] biases = new SimpleMatrix[layerNodes.length];

    static SimpleMatrix inputData = inputData();  // n0 by m
    static SimpleMatrix outputLabels = outputLabels();  // n3 by m

    // inputs: weight, height
    public static SimpleMatrix inputData() {
        SimpleMatrix out = SimpleMatrix.ones(trainingDataCount, layerNodes[0]);  // m by n0
        out.set(0, 0, 150);
        out.set(0, 1, 70);
        out.set(1, 0, 254);
        out.set(1, 1, 73);
        out.set(2, 0, 312);
        out.set(2, 1, 68);
        out.set(3, 0, 120);
        out.set(3, 1, 60);
        out.set(4, 0, 154);
        out.set(4, 1, 61);
        out.set(5, 0, 212);
        out.set(5, 1, 65);
        out.set(6, 0, 216);
        out.set(6, 1, 67);
        out.set(7, 0, 145);
        out.set(7, 1, 67);
        out.set(8, 0, 184);
        out.set(8, 1, 64);
        out.set(9, 0, 130);
        out.set(9, 1, 69);

        // find mean & deviation
        double meanWeight = 0;
        double meanHeight = 0;
        for (int i = 0; i < trainingDataCount; i++) {
            meanWeight += out.get(i, 0);
            meanHeight += out.get(i, 1);
        }
        meanWeight /= trainingDataCount;
        meanHeight /= trainingDataCount;

        double deviationWeight = 0;
        double deviationHeight = 0;
        for (int i = 0; i < trainingDataCount; i++) {
            deviationWeight += Math.pow(out.get(i, 0) - meanWeight, 2);
            deviationHeight += Math.pow(out.get(i, 1) - meanHeight, 2);
        }
        deviationWeight = Math.sqrt(deviationWeight / trainingDataCount);
        deviationHeight = Math.sqrt(deviationHeight / trainingDataCount);

        // apply standard scaling
        for (int i = 0; i < trainingDataCount; i++) {
            out.set(i * 2, (out.get(i, 0) - meanWeight) / deviationWeight);
            out.set(i * 2 + 1, (out.get(i, 1) - meanHeight) / deviationHeight);
        }
        return out;
    }

    // do they have cardiovascular disease lol
    public static SimpleMatrix outputLabels() {
        SimpleMatrix out = SimpleMatrix.ones(layerNodes[layerNodes.length-1], trainingDataCount);
        out.set(0, 0);
        out.set(1, 1);
        out.set(2, 1);
        out.set(3, 0);
        out.set(4, 0);
        out.set(5, 1);
        out.set(6, 1);
        out.set(7, 0);
        out.set(8, 1);
        out.set(9, 0);
        return out;
    }

    public static void main(String[] args) {
        setupWeightsAndBiases();
        train();

        System.out.printf("starting cost: %s%n", costHistory.getFirst());
        System.out.printf("final cost: %s [%s epochs]%n", costHistory.getLast(), EPOCHS);
    }

    public static void setupWeightsAndBiases() {
        for (int l = 1; l < layerNodes.length; l++) {
            weights[l] = MatrixUtils.randomGaussianMat(layerNodes[l], layerNodes[l-1]);
            biases[l] = MatrixUtils.broadcast(MatrixUtils.randomGaussianMat(layerNodes[l], 1), trainingDataCount);
        }
    }

    public static void train() {
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            SimpleMatrix yHat = feedForward(inputData);  // A3, the output, the prediction!

            // find the error
            double cost = calcCost(yHat);
            costHistory.add(cost);
            if (epoch % 20 == 0) System.out.printf("[%s] %s%n", epoch, cost);

            // backpropagation
            HashMap<Integer, BackPropValues> bpvs = new HashMap<>();
            bpvs.put(layerNodes.length-1, backPropOutputLayer(yHat));
            for (int l = layerNodes.length-2; l > 0; l--) {
                bpvs.put(l, backPropHiddenLayer(bpvs.get(l+1), l));
            }

            // update weights & biases
            for (int l = 1; l < layerNodes.length; l++) {
                weights[l] = weights[l].minus(bpvs.get(l).weightGradient.scale(ALPHA));
                biases[l] = biases[l].minus(MatrixUtils.broadcast(bpvs.get(l).biasGradient.scale(ALPHA), trainingDataCount));
            }
        }
    }

    public static SimpleMatrix sigmoid(SimpleMatrix mat) {
        Equation eq = new Equation();
        eq.alias(mat, "mat");
        eq.process("out = 1 / (1 + exp(-1 * mat))");
        return eq.lookupSimple("out");
    }

    public static SimpleMatrix feedForward(SimpleMatrix input) {
        activationCache.clear();
        SimpleMatrix activatedValues = input.transpose();  // from m by n0 to n0 by m
        activationCache.put(0, activatedValues);

        for (int i = 1; i < layerNodes.length; i++) {
            SimpleMatrix nodeValues = weights[i].mult(activatedValues).plus(biases[i]);
            activatedValues = sigmoid(nodeValues);
            activationCache.put(i, activatedValues);
        }
        return activatedValues;
    }

    // binary cross entropy loss summed (except i changed it, so its probably doing something else lol oops)
    public static double calcCost(SimpleMatrix yHat) {
        double summedLosses = 0;
        for (int i = 0; i < trainingDataCount; i++) {
            double yI = outputLabels.get(i);
            double yHatI = yHat.get(i);
            double loss = yI == 0 ? 1 - yHatI : yHatI;
            summedLosses -= Math.log(loss);
        }
        return (1d / trainingDataCount) * summedLosses;
    }

    // finds dC_dW[l]
    public static SimpleMatrix weightsGradient(SimpleMatrix dC_dZl, SimpleMatrix activatedValuesPrev) {
        return dC_dZl.mult(activatedValuesPrev.transpose());
    }

    // finds dC_db[l]
    public static SimpleMatrix biasesGradient(SimpleMatrix dC_dZl, int layer) {
        SimpleMatrix dC_dbl = SimpleMatrix.filled(layerNodes[layer], 1, 0);
        for (int i = 0; i < layerNodes[layer]; i++) {
            for (int j = 0; j < dC_dZl.getNumCols(); j++) {  // compress (sum) columns into one axis
                dC_dbl.set(i, dC_dbl.get(i) + dC_dZl.get(i, j));
            }
        }
        return dC_dbl;
    }

    // finds dC_dA[l-1]
    public static SimpleMatrix calcPropagator(SimpleMatrix dC_dZl, int layer) {
        return weights[layer].transpose().mult(dC_dZl);
    }

    public static BackPropValues backPropOutputLayer(SimpleMatrix yHat) {
        int layer = layerNodes.length - 1;
        SimpleMatrix activatedValuesPrev = activationCache.get(layer - 1);  // dC_AL

        Equation eq = new Equation();
        eq.alias(yHat, "yHat", outputLabels, "Y", trainingDataCount, "m");
        eq.process("out = (1.0 / m) * (yHat - Y)");
        SimpleMatrix dC_dZl = eq.lookupSimple("out");

        return new BackPropValues(
                weightsGradient(dC_dZl, activatedValuesPrev),
                biasesGradient(dC_dZl, layer),
                calcPropagator(dC_dZl, layer)
        );
    }

    public static BackPropValues backPropHiddenLayer(BackPropValues previousLayerValues, int layer) {
        SimpleMatrix activatedValues = activationCache.get(layer);  // dC_Al
        SimpleMatrix activatedValuesPrev = activationCache.get(layer - 1);  // dC_A[l-1]

        // sigmoid derivation: sigmoid'(z) = sigmoid(z) * (1-sigmoid(z))
        Equation eq = new Equation();
        eq.alias(activatedValues, "dC_Al", previousLayerValues.propagator, "dC_dAl_m1");
        eq.process("out = dC_dAl_m1 .* (dC_Al .* (1 - dC_Al))");  // .* is element-wise multiplication
        SimpleMatrix dC_dZl = eq.lookupSimple("out");

        return new BackPropValues(
                weightsGradient(dC_dZl, activatedValuesPrev),
                biasesGradient(dC_dZl, layer),
                calcPropagator(dC_dZl, layer)
        );
    }
}
import org.ejml.equation.Equation;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.HashMap;

public class Main {
    static public class BackPropValues {
        SimpleMatrix costWeights;
        SimpleMatrix costBiases;
        SimpleMatrix costPropagator;

        public BackPropValues(SimpleMatrix costWeights, SimpleMatrix costBiases, SimpleMatrix costPropagator) {
            this.costWeights = costWeights;
            this.costBiases = costBiases;
            this.costPropagator = costPropagator;
        }
    }

    static final int EPOCHS = 1000;
    static final double ALPHA = 0.1;  // learning rate
    static final ArrayList<Double> costHistory = new ArrayList<>();
    static final HashMap<String, SimpleMatrix> activationCache = new HashMap<>();

    static int trainingDataCount = 10;
    static int[] layerNodes = new int[]{2, 3, 3, 1};

    static SimpleMatrix[] weights = new SimpleMatrix[]{
            null,
            MatrixUtils.randomGaussianMat(layerNodes[1], layerNodes[0]),
            MatrixUtils.randomGaussianMat(layerNodes[2], layerNodes[1]),
            MatrixUtils.randomGaussianMat(layerNodes[3], layerNodes[2])
    };

    static SimpleMatrix[] biases = new SimpleMatrix[]{
            null,
            MatrixUtils.broadcast(MatrixUtils.randomGaussianMat(layerNodes[1], 1), trainingDataCount),
            MatrixUtils.broadcast(MatrixUtils.randomGaussianMat(layerNodes[2], 1), trainingDataCount),
            MatrixUtils.broadcast(MatrixUtils.randomGaussianMat(layerNodes[3], 1), trainingDataCount)
    };

    static SimpleMatrix inputData = inputData();  // n0 by m
    static SimpleMatrix outputLabels = outputLabels();  // n3 by m

    // inputs: weight, height
    public static SimpleMatrix inputData() {
        SimpleMatrix out = SimpleMatrix.ones(trainingDataCount, layerNodes[0]);  // m by n0
        out.set(0, 0, 150);out.set(0, 1, 70);
        out.set(1, 0, 254);out.set(1, 1, 73);
        out.set(2, 0, 312);out.set(2, 1, 68);
        out.set(3, 0, 120);out.set(3, 1, 60);
        out.set(4, 0, 154);out.set(4, 1, 61);
        out.set(5, 0, 212);out.set(5, 1, 65);
        out.set(6, 0, 216);out.set(6, 1, 67);
        out.set(7, 0, 145);out.set(7, 1, 67);
        out.set(8, 0, 184);out.set(8, 1, 64);
        out.set(9, 0, 130);out.set(9, 1, 69);

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
        SimpleMatrix out = SimpleMatrix.ones(layerNodes[3], trainingDataCount);
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
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            SimpleMatrix yHat = feedForward(inputData);  // A3, the output, the clanker's prediction!

            // find the error
            double cost = calcCost(yHat);
            costHistory.add(cost);
            if (epoch % 20 == 0) System.out.printf("[%s] %s%n", epoch, cost);

            // backpropagation
            BackPropValues bpv3 = backPropLayer3(yHat);
            BackPropValues bpv2 = backPropLayer2(bpv3);
            BackPropValues bpv1 = backPropLayer1(bpv2);

            // update weights
            Equation eqW = new Equation();
            eqW.alias(ALPHA, "a", bpv3.costWeights, "w3", bpv2.costWeights, "w2", bpv1.costWeights, "w1");
            eqW.process("w3_out = a * w3");
            eqW.process("w2_out = a * w2");
            eqW.process("w1_out = a * w1");
            weights[3] = weights[3].minus(eqW.lookupSimple("w3_out"));
            weights[2] = weights[2].minus(eqW.lookupSimple("w2_out"));
            weights[1] = weights[1].minus(eqW.lookupSimple("w1_out"));

            // update biases
            Equation eqB = new Equation();
            eqB.alias(ALPHA, "a", bpv3.costBiases, "b3", bpv2.costBiases, "b2", bpv1.costBiases, "b1");
            eqB.process("b3_out = a * b3");
            eqB.process("b2_out = a * b2");
            eqB.process("b1_out = a * b1");
            biases[3] = biases[3].minus(MatrixUtils.broadcast(eqB.lookupSimple("b3_out"), trainingDataCount));
            biases[2] = biases[2].minus(MatrixUtils.broadcast(eqB.lookupSimple("b2_out"), trainingDataCount));
            biases[1] = biases[1].minus(MatrixUtils.broadcast(eqB.lookupSimple("b1_out"), trainingDataCount));
        }
        System.out.printf("final cost: %s%n", costHistory.getLast());
    }

    public static SimpleMatrix sigmoid(SimpleMatrix mat) {
        Equation eq = new Equation();
        eq.alias(mat, "mat");
        eq.process("out = 1 / (1 + exp(-1 * mat))");
        return eq.lookupSimple("out");
    }

    public static SimpleMatrix feedForward(SimpleMatrix input) {
        SimpleMatrix A0 = input.transpose();  // from m by n0 to n0 by m

        SimpleMatrix Z1 = weights[1].mult(A0).plus(biases[1]);
        SimpleMatrix A1 = sigmoid(Z1);

        SimpleMatrix Z2 = weights[2].mult(A1).plus(biases[2]);
        SimpleMatrix A2 = sigmoid(Z2);

        SimpleMatrix Z3 = weights[3].mult(A2).plus(biases[3]);
        SimpleMatrix A3 = sigmoid(Z3);

        activationCache.clear();
        activationCache.put("A0", A0);
        activationCache.put("A1", A1);
        activationCache.put("A2", A2);
        return A3;
    }

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

    public static BackPropValues backPropLayer3(SimpleMatrix A3) {
        SimpleMatrix A2 = activationCache.get("A2");
        SimpleMatrix W3 = weights[3];

        Equation eq = new Equation();
        eq.alias(A3, "A3", outputLabels, "Y", trainingDataCount, "m");
        eq.process("out = (1.0 / m) * (A3 - Y)");
        SimpleMatrix dC_dZ3 = eq.lookupSimple("out");

        // weights
        SimpleMatrix dC_dW3 = dC_dZ3.mult(A2.transpose());

        // biases
        SimpleMatrix dC_db3 = SimpleMatrix.filled(layerNodes[3], 1, 0);
        for (int i = 0; i < layerNodes[3]; i++) {
            for (int j = 0; j < dC_dZ3.getNumCols(); j++) {  // compress (sum) columns into one axis
                dC_db3.set(i, dC_db3.get(i) + dC_dZ3.get(i, j));
            }
        }

        // for continuing the chain to the next layer, the propagator
        SimpleMatrix dC_dA2 = W3.transpose().mult(dC_dZ3);
        return new BackPropValues(dC_dW3, dC_db3, dC_dA2);
    }

    public static BackPropValues backPropLayer2(BackPropValues bpv3) {
        SimpleMatrix A1 = activationCache.get("A1");
        SimpleMatrix A2 = activationCache.get("A2");
        SimpleMatrix W2 = weights[2];

        // sigmoid derivation ( sigmoid'(z) = sigmoid(z) * (1-sigmoid(z)) )
        Equation eq = new Equation();
        eq.alias(A2, "A2", bpv3.costPropagator, "dC_dA2");
        eq.process("out = dC_dA2 .* (A2 .* (1 - A2))");  // .* is element-wise multiplication
        SimpleMatrix dC_dZ2 = eq.lookupSimple("out");

        // weights
        SimpleMatrix dC_dW2 = dC_dZ2.mult(A1.transpose());

        // biases
        SimpleMatrix dC_db2 = SimpleMatrix.filled(layerNodes[2], 1, 0);
        for (int i = 0; i < layerNodes[2]; i++) {
            for (int j = 0; j < dC_dW2.getNumCols(); j++) {  // compress (sum) columns into one axis
                dC_db2.set(i, dC_db2.get(i) + dC_dW2.get(i, j));
            }
        }

        // propagator
        SimpleMatrix dC_dA1 = W2.transpose().mult(dC_dZ2);
        return new BackPropValues(dC_dW2, dC_db2, dC_dA1);
    }

    public static BackPropValues backPropLayer1(BackPropValues bpv2) {
        SimpleMatrix A0 = activationCache.get("A0");
        SimpleMatrix A1 = activationCache.get("A1");
        SimpleMatrix W1 = weights[1];

        // sigmoid derivation ( sigmoid'(z) = sigmoid(z) * (1-sigmoid(z)) )
        Equation eq = new Equation();
        eq.alias(A1, "A1", bpv2.costPropagator, "dC_dA1");
        eq.process("out = dC_dA1 .* (A1 .* (1 - A1))");  // .* is element-wise multiplication
        SimpleMatrix dC_dZ1 = eq.lookupSimple("out");

        // weights
        SimpleMatrix dC_dW1 = dC_dZ1.mult(A0.transpose());

        // biases
        SimpleMatrix dC_db1 = SimpleMatrix.filled(layerNodes[1], 1, 0);
        for (int i = 0; i < layerNodes[2]; i++) {
            for (int j = 0; j < dC_dW1.getNumCols(); j++) {  // compress (sum) columns into one axis
                dC_db1.set(i, dC_db1.get(i) + dC_dW1.get(i, j));
            }
        }

        // no propagator needed, this is last layer
        return new BackPropValues(dC_dW1, dC_db1, null);
    }
}
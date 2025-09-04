import org.ejml.equation.Equation;
import org.ejml.simple.SimpleMatrix;

public class Main {
    static int m = 10;
    static int[] n = new int[]{2, 3, 3, 1};

    static SimpleMatrix[] W = new SimpleMatrix[]{
            null,
            MatrixUtils.randomGaussianMat(n[1], n[0]),
            MatrixUtils.randomGaussianMat(n[2], n[1]),
            MatrixUtils.randomGaussianMat(n[3], n[2])
    };

    static SimpleMatrix[] b = new SimpleMatrix[]{
            null,
            MatrixUtils.broadcast(MatrixUtils.randomGaussianMat(n[1], 1), m),
            MatrixUtils.broadcast(MatrixUtils.randomGaussianMat(n[2], 1), m),
            MatrixUtils.broadcast(MatrixUtils.randomGaussianMat(n[3], 1), m)
    };

    // input data
    static SimpleMatrix X = inputData();  // m by n0
    static SimpleMatrix A0 = X.transpose();  // A^[0] `n0 by m`

    // training labels
    static SimpleMatrix y = trainingData();  // n3 by m
    static SimpleMatrix Y = y.transpose();  // m by n3

    public static SimpleMatrix inputData() {
        SimpleMatrix out = SimpleMatrix.ones(m, n[0]);
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
        return out;
    }

    public static SimpleMatrix trainingData() {
        SimpleMatrix out = SimpleMatrix.ones(n[3], m);
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
        SimpleMatrix yHat = feedForward();
        double cost = calcCost(yHat);
        System.out.println(cost);
    }

    public static SimpleMatrix sigmoid(SimpleMatrix mat) {
        Equation eq = new Equation();
        eq.alias(mat, "mat");
        eq.process("out = 1 / (1 + exp(-1 * mat))");
        return eq.lookupSimple("out");
    }

    public static SimpleMatrix feedForward() {
        SimpleMatrix Z1 = W[1].mult(A0).plus(b[1]);
        SimpleMatrix A1 = sigmoid(Z1);

        SimpleMatrix Z2 = W[2].mult(A1).plus(b[2]);
        SimpleMatrix A2 = sigmoid(Z2);

        SimpleMatrix Z3 = W[3].mult(A2).plus(b[3]);
        SimpleMatrix A3 = sigmoid(Z3);

        System.out.println(A3);
        MatrixUtils.printMatrixArray(A3.toArray2());
        return A3;
    }

    public static double calcCost(SimpleMatrix yHat) {
        double summed = 0;
        for (int i = 0; i < m; i++) {
            double yI = Y.get(i);
            double yHatI = yHat.get(i);
            double c = yI == 0 ? 1 - yHatI : yHatI;
//            if (yI == 0) {
//                c = 1 - yHatI;
//            } else {
//                c = yHatI;
//            }
            summed -= Math.log(c);
//            System.out.println(yI);
//            System.out.println(yHatI);
//            double c = yI * Math.log(yHatI + (1 - yI) * Math.log(1 - yHatI));
//            System.out.println(c);
//            summed -= c;
        }
//        System.out.println(summed);
        //        Equation eq = new Equation();
//        eq.alias(yHat, "yHat", Y, "Y");
//        eq.process("out = -( (Y * log(yHat) + (1 - Y)*log(1 - yHat)) )");
//        SimpleMatrix losses = eq.lookupSimple("out");
//        SimpleMatrix m = losses.reshape(-1);
//        double summed =
        return (1d / m) * summed;
    }

    public static void backProp(double cost) {

    }
}
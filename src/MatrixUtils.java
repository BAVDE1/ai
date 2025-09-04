import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class MatrixUtils {
    public static SimpleMatrix broadcast(SimpleMatrix mat, int count) {
        SimpleMatrix out = SimpleMatrix.ones(mat.getNumRows(), count);
        for (int i = 0; i < mat.getNumRows(); i++) {
            for (int j = 0; j < count; j++) out.set(i, j, mat.get(i, 0));
        }
        return out;
    }

    public static SimpleMatrix randomGaussianMat(int rows, int cols) {
        SimpleMatrix out = SimpleMatrix.ones(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) out.set(i, j, new Random().nextGaussian());
        }
        return out;
    }

    public static void printMatrixArray(double[][] array) {
        for (double[] row : array) {
            for (double val : row) System.out.print(val + " ");
            System.out.print("\n");
        }
    }

    public static void printShape(int[][] array) {
        System.out.printf("%s x %s%n", array.length, array[0].length);
    }
}

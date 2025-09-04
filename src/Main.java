import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;

public class Main {
    static int[] n = new int[] {2, 3, 3, 1};
    static SimpleMatrix[] W = new SimpleMatrix[] {
            SimpleMatrix.random(n[1], n[0]),
            SimpleMatrix.random(n[2], n[1]),
            SimpleMatrix.random(n[3], n[2])
    };
    static SimpleMatrix[] b = new SimpleMatrix[] {
            SimpleMatrix.random(n[1], 1),
            SimpleMatrix.random(n[2], 1),
            SimpleMatrix.random(n[3], 1)
    };

    public static void main(String[] args) {
        System.out.println(Arrays.toString(W));
        System.out.println(Arrays.toString(b));
    }
}
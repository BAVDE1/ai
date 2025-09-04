import java.util.Arrays;

public class Main {
    static int[] n = new int[] {2, 3, 3, 1};
    static Matrix[] W = new Matrix[] {
            new Matrix(n[1], n[0]).randomise(),
            new Matrix(n[2], n[1]).randomise(),
            new Matrix(n[3], n[2]).randomise()
    };
    static Matrix[] b = new Matrix[] {
            new Matrix(n[1], 1).randomise(),
            new Matrix(n[2], 1).randomise(),
            new Matrix(n[3], 1).randomise()
    };

    public static void main(String[] args) {
        System.out.println(Arrays.toString(W));
        System.out.println(Arrays.toString(b));
    }
}
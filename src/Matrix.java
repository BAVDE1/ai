import java.util.Arrays;
import java.util.Random;

public class Matrix {
    final int rows;
    final int columns;

    final double[][] contents;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;

        this.contents = new double[rows][columns];
    }

    public Matrix randomise() {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                contents[r][c] = new Random().nextGaussian();
            }
        }
        return this;
    }

    @Override
    public String toString() {
        String s = "";
        for (double[] row : contents) s = s.concat(Arrays.toString(row));
        return "Matrix(%s x %s: %s)".formatted(rows, columns, s);
    }
}

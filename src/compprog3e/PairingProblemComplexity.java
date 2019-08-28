package compprog3e;

import java.math.BigInteger;
import java.text.NumberFormat;
import java.util.stream.IntStream;

public class PairingProblemComplexity {
    public static void main(String[] args) {
        System.out.println(String.format("%10s\t%20s\t%20s\t%20s",
            "NUM POINTS", "ALL ORDERED", "IGNORE WITHIN PAIRS", "IGNORE ALL"));
        NumberFormat formatter = NumberFormat.getIntegerInstance();
        IntStream.iterate(2, i -> i + 2).limit(9).forEach(i ->
            System.out.println(String.format("%10d\t%20s\t%20s\t%20s",
                i,
                formatter.format(factorial(i)),
                formatter.format(withinPairs(i)),
                formatter.format(noRegard(i))))
        );
    }

    static BigInteger factorial(int n) {
        BigInteger fac = BigInteger.ONE;
        for (int i = 2; i <= n; i++) {
            fac = fac.multiply(BigInteger.valueOf(i));
        }
        return fac;
    }

    static BigInteger withinPairs(int n) {
        BigInteger ret = factorial(n);
        for (int i = 0; i < n/2; i++) {
            ret = ret.divide(BigInteger.TWO);
        }
        return ret;
    }

    static BigInteger noRegard(int n) {
        BigInteger ret = withinPairs(n);
        return ret.divide(factorial(n/2));
    }
}

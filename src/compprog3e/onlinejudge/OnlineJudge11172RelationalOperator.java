package compprog3e.onlinejudge;

import java.util.Scanner;

// Only interesting because a Java integer is always 32 bits
// and uses two's complement encoding, thus covering the range
// from -2^32 to 2^32 - 1, inclusive (roughly +/- 2.1 billion).
public class OnlineJudge11172RelationalOperator {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int numCases = scanner.nextInt();
        for (int i = 0; i < numCases; i++) {
            int a = scanner.nextInt();
            int b = scanner.nextInt();
            System.out.println(a < b ? "<" : a > b ? ">" : "=");
        }
    }
}

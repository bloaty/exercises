package compprog3e.onlinejudge;

import java.util.Arrays;
import java.util.IntSummaryStatistics;
import java.util.Scanner;

// Small enough to do a linear search.
public class OnlineJudge11364Parking {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int numCases = scanner.nextInt();
        for (int i = 0; i < numCases; i++) {
            int numLocations = scanner.nextInt();
            int[] positions = new int[numLocations];
            for (int j = 0; j < numLocations; j++) {
                positions[j] = scanner.nextInt();
            }
            int bestSum = Integer.MAX_VALUE;
            IntSummaryStatistics inputStats = Arrays.stream(positions).summaryStatistics();
            int max = inputStats.getMax();
            int min = inputStats.getMin();
            for (int k = 0; k < 100; k++) {
                int sum;
                if (k < min) {
                    sum = 2 * (max - k);
                } else if (k > max) {
                    sum = 2 * (k - min);
                } else {
                    sum = 2 * (max - min);
                }
                if (sum < bestSum) bestSum = sum;
            }
            System.out.println(bestSum);
        }
    }
}

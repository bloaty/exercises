package compprog3e.onlinejudge;

import java.util.Scanner;

public class OnlineJudge11044SearchingForNessy {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int numCases = scanner.nextInt();
        for (int i = 0; i < numCases; i++) {
            System.out.println(numSonars(scanner.nextInt(), scanner.nextInt()));
        }
    }

    // Perhaps only notable because integer division is not commutative.
    private static int numSonars(int rows, int columns) {
        return (rows / 3) * (columns / 3);
    }
}

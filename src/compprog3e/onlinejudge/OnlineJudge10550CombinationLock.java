package compprog3e.onlinejudge;

import java.util.Scanner;

// This one proved strangely tricky. I kept forgetting
// that rotating the dial clockwise effectively moves
// the position indicator counterclockwise.
public class OnlineJudge10550CombinationLock {
    private static final String LIMIT = "0 0 0 0";
    public static void main(String[] args) {
        String line;
        try (Scanner scanner = new Scanner(System.in)) {
            while (scanner.hasNextLine()) {
                line = scanner.nextLine();
                if (line.contains(LIMIT)) break;
                String[] tokens = line.trim().split("\\s+");
                int i = Integer.parseInt(tokens[0]);
                int a = Integer.parseInt(tokens[1]);
                int b = Integer.parseInt(tokens[2]);
                int c = Integer.parseInt(tokens[3]);
                int ticks = 40 + 40 + (i >= a ? i - a : 40 - (a - i))
                    + 40 + (b >= a ? b - a : 40 - (a - b))
                    + (b >= c ? b - c : 40 - (c - b));
                int degrees = ticks * 9;
                System.out.println(degrees);
            }
        }
    }
}

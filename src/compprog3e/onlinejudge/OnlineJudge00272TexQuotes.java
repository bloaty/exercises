package compprog3e.onlinejudge;

import java.util.Scanner;

public class OnlineJudge00272TexQuotes {

    private static final String TICKS = "``";
    private static final String QUOTES = "\'\'";

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        boolean opening = true;
        String line;
        StringBuffer buf;
        while (scanner.hasNextLine()) {
            line = scanner.nextLine();
            buf = new StringBuffer();
            char[] chars = line.toCharArray();
            for (int i = 0; i < chars.length; i++) {
                if (chars[i] == '"') {
                    buf.append(opening ? TICKS : QUOTES);
                    opening = !opening;
                } else {
                    buf.append(chars[i]);
                }
            }
            System.out.println(buf.toString());
        }
    }
}

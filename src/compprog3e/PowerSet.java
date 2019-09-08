package compprog3e;

import java.util.Arrays;
import java.util.stream.LongStream;

public class PowerSet {
    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("[]");
            return;
        }
        String[] items = args;
        int numItems = items.length;
        int outputCharLength = Arrays.stream(args).mapToInt(String::length).sum() + 2 + (numItems - 1) * 2;
        LongStream.range(0, 1 << numItems)
            .mapToObj(i -> PowerSet.longToString(i, items, outputCharLength))
            .forEach(System.out::println);
    }

    static String longToString(long aLong, String[] items, int outputCharLength) {
        StringBuilder buf = new StringBuilder(outputCharLength);
        buf.append('[');
        int lengthInBinaryDigits = Long.SIZE - Long.numberOfLeadingZeros(aLong);
        for (int i = 0; i < lengthInBinaryDigits; i++) {
            if ((aLong & (1 << i)) != 0) {
                buf.append(items[i]);
                buf.append(' ');
            }
        }
        return buf.toString().trim() + ']';
    }
}

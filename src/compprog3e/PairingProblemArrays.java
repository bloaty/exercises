package compprog3e;

import java.awt.geom.Point2D;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Double.parseDouble;

public final class PairingProblemArrays {
    public static void main(String[] args) throws IOException {
        // Read file input.
        Path inputFile = Path.of(args[0]);
        Stream<String> linesFromFile = Files.newBufferedReader(inputFile)
            .lines()
            .filter(s -> !s.isEmpty());

        // Parse file input. Example input:
        // 1 , 2 \n 3 , 5 \n 0 , -4
        Function<String, Point2D.Double> parsePointFromLine = line -> {
            String[] tokens = line.split("\\s*,\\s*");
            double x = parseDouble(tokens[0]);
            double y = parseDouble(tokens[1]);
            return new Point2D.Double(x, y);
        };

        Point2D.Double[] points = linesFromFile.map(parsePointFromLine)
            .toArray(Point2D.Double[]::new);

        if (points.length % 2 != 0)
            throw new RuntimeException("Must have an even number of points!");

        // Compute pairwise distances.
        int numPoints = points.length;
        double[][] distances = new double[numPoints][numPoints];

        for (int i = 0; i < numPoints; i++) {
            for (int j = i + 1; j < numPoints; j++) {
                double iToJ = Math.abs(Math.hypot(
                    points[i].x - points[j].x,
                    points[i].y - points[j].y));
                distances[i][j] = iToJ;
                distances[j][i] = iToJ;
            }
        }

        // Compute all unique pairings (i.e., without regard for the
        // order of pairs in a pairing or points in a given pair).
        int[] initialIndexes = IntStream.range(0, numPoints).toArray();
        Stream<int[]> pairings = matchUp(initialIndexes, 0, numPoints);

        ConcurrentLinkedQueue<Pairing> bests = new ConcurrentLinkedQueue<>();
        // Initialize so that we don't have to check for empty every time.
        bests.add(Pairing.THE_WORST_PAIRING);
        pairings.forEach(pairing -> {
            double pathSum = Pairing.calculatePathSum(pairing, distances);
            double currentBest = bests.peek().totalDistance;
            if (pathSum < currentBest) {
                bests.clear();
                bests.add(new Pairing(pairing, distances, points));
                return;
            }
            if (pathSum == currentBest) {
                bests.add(new Pairing(pairing, distances, points));
            }
        });

        // Print all winners to STDOUT.
        bests.forEach(System.out::println);
    }

    // Each int[] is a full pairing and has a length equal to the number of points.
    // Successive pairs of entries are, implicitly, indexes of paired points,
    // namely, { p0_0, p0_1, p1_0, p1_1, p2_0, ... }. At a depth of N, this method
    // sets the N-th pair at indexes (N * 2) and (N * 2 + 1).
    // We stream the pairings so that we don't have to have an exhaustive
    // enumeration of them in memory at once.
    private static Stream<int[]> matchUp(int[] indexes, int depth, int numPoints) {
        if (indexes.length == 0) {
            return Stream.empty();
        } else if (indexes.length == 2) {
            int[] pairs = new int[numPoints];
            Arrays.fill(pairs, -1);
            pairs[depth * 2] = indexes[0];
            pairs[depth * 2 + 1] = indexes[1];
            return Stream.of(pairs);
        } else {
            // This is the non-trivial case. We build every pair between the first remaining
            // index and every other remaining index. For each choice of pair, we exclude the
            // indices we selected and recurse into the remaining indices.
            // We do not build every possible pair, because then we will generate pairings
            // that differ only in the order of the pairs. Namely, given [1, 2, 3, 4], if we
            // build only [1, 2] at depth 0, the only choice is [3, 4] at depth 1. If we build
            // both [1, 2] and [3, 4] at depth 0, we then build both complements at depth 1 and
            // end up with two equivalent pairings, [[1, 2], [3, 4]] and [[3, 4], [1, 2]].
            Stream<int[]> pairings = Stream.empty();
            for (int i = 1; i < indexes.length; i++) {
                int[] newIndexes = new int[indexes.length - 2];
                for (int j = 1, k = 0; j < indexes.length; j++) {
                    if (j != i) {
                        newIndexes[k++] = indexes[j];
                    }
                }
                final int p1 = indexes[0];
                final int p2 = indexes[i];
                Stream<int[]> partialPairings = matchUp(newIndexes, depth + 1, numPoints);
                partialPairings = partialPairings.peek(pairing -> {
                        pairing[depth * 2] = p1;
                        pairing[depth * 2 + 1] = p2;
                    });
                pairings = Stream.of(pairings, partialPairings).flatMap(x -> x);
            }
            return pairings;
        }
    }

    private static class Pairing {
        int[] pairing;
        double totalDistance;
        Point2D.Double[] points;

        static final Pairing THE_WORST_PAIRING = new Pairing();

        Pairing(int[] pairing, double[][] distances, Point2D.Double[] points) {
            this.pairing = pairing;
            this.totalDistance = calculatePathSum(pairing, distances);
            this.points = points;
        }

        Pairing() {
            this.pairing = new int[0];
            this.totalDistance = Double.MAX_VALUE;
            this.points = new Point2D.Double[0];
        }

        static double calculatePathSum(int[] pairing, double[][] distances) {
            double sum = 0;
            for (int i = 0; i < pairing.length; i = i + 2) {
                sum += distances[pairing[i]][pairing[i + 1]];
            }
            return sum;
        }

        @Override
        public String toString() {
            List<String> pairs = new ArrayList<>();
            for (int i = 0; i < pairing.length; i = i + 2) {
                pairs.add(String.format(
                    "[(%.1f, %.1f), (%.1f, %.1f)]",
                    points[pairing[i]].getX(),
                    points[pairing[i]].getY(),
                    points[pairing[i + 1]].getX(),
                    points[pairing[i + 1]].getY()));
            }
            return String.join(" , ", pairs) + " : total distance = " + totalDistance;
        }
    }
}

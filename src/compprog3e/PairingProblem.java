package compprog3e;

import java.awt.geom.Point2D;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Double.parseDouble;

public class PairingProblem {
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
        Stream<List<int[]>> pairings = matchUp(initialIndexes);

        Pairing worst = new Pairing(new ArrayList<>(), Double.MAX_VALUE, distances, points);
        ConcurrentLinkedQueue<Pairing> bests = new ConcurrentLinkedQueue<>();
        bests.add(worst);
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
        bests.forEach(System.out::println);
    }

    // Each int[] is a 2-array of int representing two paired-up points.
    // For example, {3, 5} is a pair of point #3 and point #5).
    // A List<int[]> is a collection of pairs that we are building into a
    // complete pairing. Namely, if there are 2N points, a complete pairing
    // contains N pairs without any repeated points.
    // We stream the points so that we don't have to have an exhaustive
    // enumeration of them in memory at once.
    private static Stream<List<int[]>> matchUp(int[] indexes) {
        if (indexes.length == 0) {
            return Stream.empty();
        } else if (indexes.length == 2) {
            List<int[]> pairs = new ArrayList<>();
            pairs.add(new int[] {indexes[0], indexes[1]});
            return Stream.of(pairs);
        } else {
            // This is the non-trivial case. We build every pair between the first remaining
            // index and every other remaining index. For each of these pairs, we exclude the
            // indices we selected and recurse into the remaining indices.
            // We do not build every possible pair, because then we will generate pairings
            // that differ only in the order of the pairs. Namely, given [1, 2, 3, 4], if we
            // build only [1, 2] in pass 1, the only choice is [3, 4] in pass 2. If we build
            // both [1, 2] and [3, 4] in pass 1, we then build both complements in pass 2 and end
            // up with two equivalent pairings, [[1, 2], [3, 4]] and [[3, 4], [1, 2]].
            Stream<List<int[]>> pairings = Stream.empty();
            for (int i = 1; i < indexes.length; i++) {
                int[] newIndexes = new int[indexes.length - 2];
                for (int j = 1, k = 0; j < indexes.length; j++) {
                    if (j != i) {
                        newIndexes[k++] = indexes[j];
                    }
                }
                int[] pair = new int[] {indexes[0], indexes[i]};
                Stream<List<int[]>> suffixes = matchUp(newIndexes).peek(x -> x.add(pair));
                pairings = Stream.concat(pairings, suffixes);
            }
            return pairings;
        }
    }

    private static class Pairing {
        List<int[]> pairing;
        double totalDistance;
        Point2D.Double[] points;

        Pairing(List<int[]> pairing, double[][] distances, Point2D.Double[] points) {
            this.pairing = pairing;
            this.totalDistance = calculatePathSum(pairing, distances);
            this.points = points;
        }

        Pairing(List<int[]> pairing, double totalDistance, double[][] distances, Point2D.Double[] points) {
            this.pairing = pairing;
            this.totalDistance = calculatePathSum(pairing, distances);
            this.totalDistance = totalDistance;
            this.points = points;
        }

        static double calculatePathSum(List<int[]> pairing, double[][] distances) {
            return pairing.stream()
                .mapToDouble(pair -> distances[pair[0]][pair[1]])
                .sum();
        }

        @Override
        public String toString() {
            String pairs = pairing.stream()
                .map(pair -> String.format(
                    "[(%.1f, %.1f), (%.1f, %.1f)]",
                    points[pair[0]].getX(),
                    points[pair[0]].getY(),
                    points[pair[1]].getX(),
                    points[pair[1]].getY()))
                .collect(Collectors.joining(" , "));
            return pairs + " : total distance = " + totalDistance;
        }
    }
}

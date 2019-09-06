package adm2e.tsp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

/**
 * A naive greedy TSP that starts from a random solution
 * and makes the most advantageous pairwise vertex swap
 * until it gets stuck in a local minimum.
 */
public class LocalSearch {

    private static void usage() {
        System.err.println(
            "Usage: java -cp 'out:lib' "
            + LocalSearch.class.getName()
            + " <path to input file> <number of retries>");
    }

    public static void main(String[] args) throws IOException {
        // Get input.
        if (args.length != 2) {
            usage();
            return;
        }
        Stream<String> lines = readRawInput(args[0]);
        int numRetries = Integer.parseInt(args[1]);

        // Initialize collections for parsing input.
        Map<String, Map<String, Double>> labelToLabelToDistance = new HashMap<>();
        Set<String> labelCollector = new HashSet<>();

        // Populate collections from input.
        lines.forEach(line ->
            processLine(line, labelCollector, labelToLabelToDistance));

        // Reshape collected input.
        String[] labels = labelCollector.toArray(new String[0]);
        Arrays.sort(labels);
        double[][] distances = buildEdgeWeightMatrix(labels, labelToLabelToDistance);

        // Run solver.
        HeuristicTspSolver solver = HeuristicTspSolver.create(labels, distances);
        double bestCost = Double.MAX_VALUE;
        TspSolution bestSolution = null;
        for (int i = 0; i < numRetries; i++) {
            TspSolution solution = solver.getFixedPointSolution();
            solver = solver.reinitializedCopy();
            if (solution.getCost() < bestCost) {
                bestCost = solution.getCost();
                bestSolution = solution;
            }
        }
        System.out.println("Best solution is "
            + Arrays.toString(bestSolution.getVertexVisitOrder())
            + " with total cost "
            + bestSolution.getCost()
            + ".");
    }

    private static Stream<String> readRawInput(String path) throws IOException {
        Path inputFile = Path.of(path);
        return Files.newBufferedReader(inputFile)
                    .lines()
                    .filter(s -> !s.isEmpty());
    }

    private static void processLine(String line,
                                    Set<String> labelCollector,
                                    Map<String, Map<String, Double>> distances) {
        // Tokenize line on whitespace.
        String[] tokens = line.split("\\s+");
        // Select TO and FROM so that FROM < TO lexicographically.
        // This avoids having both A -> B and B -> A in the mapping.
        String from = tokens[0].compareTo(tokens[1]) < 0
            ? tokens[0]
            : tokens[1];
        String to = tokens[0].compareTo(tokens[1]) < 0
            ? tokens[1]
            : tokens[0];
        Double distance = Double.parseDouble(tokens[2]);
        // Initialize bucket when needed.
        if (!distances.containsKey(from)) {
            distances.put(from, new HashMap<>());
        }
        // Throw if there are multiple records for A <-> B and they disagree.
        if (distances.get(from).containsKey(to)
            && !distances.get(from).get(to).equals(distance)) {
            throw new RuntimeException(
                "Encountered conflicting records for the edge between "
                    + from + " and " + to + "!");
        }

        distances.get(from).put(to, distance);
        labelCollector.add(from);
        labelCollector.add(to);
    }

    private static double[][] buildEdgeWeightMatrix(String[] labels,
                                                    Map<String, Map<String, Double>> distances) {
        int numVertices = labels.length;
        double[][] edgeWeightMatrix = new double[numVertices][numVertices];
        for (int i = 0; i < numVertices - 1; i++) {
            for (int j = i + 1; j < numVertices; j++) {
                if (!distances.containsKey(labels[i])
                    || !distances.get(labels[i]).containsKey(labels[j])) {
                    throw new RuntimeException(
                        "Could not find distance between "
                            + labels[i] + " and " + labels[j] + "!");
                }
                edgeWeightMatrix[i][j] = distances.get(labels[i]).get(labels[j]);
                edgeWeightMatrix[j][i] = edgeWeightMatrix[i][j];
            }
        }
        return edgeWeightMatrix;
    }

    /**
     * Stores the search space for a TSP problem.
     */
    public static final class TspContext {
        private final String[] vertexLabels;
        private final double[][] vertexDistances;

        private TspContext(String[] vertexLabels, double[][] vertexDistances) {
            if (vertexLabels.length != vertexDistances.length
                || vertexDistances.length != vertexDistances[0].length) {
                throw new IllegalArgumentException(
                    "Given N vertices, the matrix of distances between them should be NxN.");
            }

            int numVertices = vertexLabels.length;
            this.vertexLabels = Arrays.copyOf(vertexLabels, numVertices);

            this.vertexDistances = new double[numVertices][numVertices];
            for (int i = 0; i < numVertices; i++) {
                for (int j = i + 1; j < numVertices; j++) {
                    this.vertexDistances[i][j] = vertexDistances[i][j];
                    this.vertexDistances[j][i] = vertexDistances[i][j];
                }
            }
        }

        public int getNumVertices() {
            return vertexLabels.length;
        }

        public String[] getVertexLabels() {
            return Arrays.copyOf(vertexLabels, vertexLabels.length);
        }

        public String getVertexLabel(int i) {
            return vertexLabels[i];
        }

        /**
         * Edge weight between two vertices.
         */
        public double getDistance(int i, int j) {
            return vertexDistances[i][j];
        }

        private double getPathCost(int[] vertexVisitOrder) {
            double cost = 0;
            // Add up distances between successive visited vertices.
            for (int i = 0; i < vertexVisitOrder.length - 1; i++) {
                cost += getDistance(vertexVisitOrder[i], vertexVisitOrder[i + 1]);
            }
            // Add final leg between last visited vertex and initial vertex.
            cost += getDistance(vertexVisitOrder[getNumVertices() - 1],
                                vertexVisitOrder[0]);
            return cost;
        }
    }

    public static final class HeuristicTspSolver {

        private final TspContext context;
        private int[] currentSolution;
        private boolean reachedFixedPoint;

        private HeuristicTspSolver(TspContext context) {
            this.context = context;
            this.currentSolution = randomPermutation(context);
            this.reachedFixedPoint = false;
        }

        public static HeuristicTspSolver create(String[] labels, double[][] distances) {
            return new HeuristicTspSolver(new TspContext(labels, distances));
        }

        /**
         * Create a new solver, which is not yet stuck in a fixed point,
         * from the same context, but with a new initial position.
         */
        public HeuristicTspSolver reinitializedCopy() {
            return new HeuristicTspSolver(this.context);
        }

        // Initialize search with a random solution -- a more or less
        // uniformly random permutation of N vertices. We make a list
        // of longs whose last 3 decimal digits are 0, add the vertex
        // indices, sort the longs, and recover the vertex indices
        // using modulo division.
        private static int[] randomPermutation(TspContext context) {
            int numVertices = context.getNumVertices();
            Random random = ThreadLocalRandom.current();
            int[] path = new int[numVertices];
            long[] randomizer = new long[numVertices];
            for (int i = 0; i < numVertices; i++) {
                randomizer[i] = 1000L * random.nextInt(Integer.MAX_VALUE) + i;
            }
            Arrays.sort(randomizer);
            for (int i = 0; i < numVertices; i++) {
                path[i] = (int) (randomizer[i] % 1000);
            }
            return path;
        }

        /**
         * Runs this search instance until it reaches a fixed point.
         * and further iteration becomes fruitless.
         * @return the solution representing the local min
         *         found by this search attempt
         */
        public TspSolution getFixedPointSolution() {
            while (!reachedFixedPoint) iterateOnce();
            return new TspSolution(context, currentSolution);
        }

        /**
         * Runs one iteration of the heuristic. If this solver has already
         * reached its local min, it just returns that without doing any work.
         */
        public TspSolution getNextSolution() {
            iterateOnce();
            return new TspSolution(context, currentSolution);
        }

        /**
         * Returns true when this solver gets stuck in a local min and
         * no longer makes progress.
         */
        public boolean reachedFixedPoint() {
            return reachedFixedPoint;
        }

        private void doActionForAllPairsOfEdges(int[] bestReachableSolution,
                                                AtomicReference<Double> bestReachableCost,
                                                SolverAction action) {
            int numVertices = context.getNumVertices();
            // First, edges <0, 1> and <2, 3>, <0, 1> and <4, 5>, ..., <2, 3> and <4, 5>, etc.
            for (int i = 0, j = 1;
                 j < numVertices - 2;
                 i += 2, j += 2) {

                for (int k = i + 2, m = j + 2;
                     m < numVertices;
                     k += 2, m += 2) {

                    action.perform(bestReachableSolution, bestReachableCost, i, j, k, m);
                }
            }
            // Then, edges <1, 2> and <3, 4>, <1, 2> and <5, 6>, ..., <3, 4> and <5, 6>, etc.
            for (int i = 1, j = 2;
                 j < numVertices - 2;
                 i += 2, j += 2) {

                for (int k = i + 2, m = j + 2;
                     m < numVertices;
                     k += 2, m += 2) {

                    action.perform(bestReachableSolution, bestReachableCost, i, j, k, m);
                }
            }
            // Finally, edge <0, LAST> and <1, 2>, <0, LAST> and <3, 4>, etc.
            for (int i = 0, j = numVertices - 1, k = 1, m = 2;
                 m < numVertices - 1;
                 k += 2, m += 2) {

                action.perform(bestReachableSolution, bestReachableCost, i, j, k, m);
            }
        }

        // Just a simple functional interface to type-alias this function signature.
        @FunctionalInterface
        private interface SolverAction {
            void perform(int[] bestReachableSolution,
                         AtomicReference<Double> bestReachableCost,
                         int i, int j, int k, int m);
        }

        private void singleMoveInSearchSpace(int[] bestReachableSolution,
                                             AtomicReference<Double> bestReachableCost,
                                             int i, int j, int k, int m) {
            swap(currentSolution, i, j, k, m);
            double newCost = context.getPathCost(currentSolution);
            if (newCost < bestReachableCost.get()) {
                bestReachableCost.set(newCost);
                System.arraycopy(currentSolution, 0, bestReachableSolution, 0, context.getNumVertices());
            }
            unswap(currentSolution, i, j, k, m);
        }

        // Edges <i, j> and <k, m> turn into <i, k> and <j, m>.
        // This only requires swapping k and j.
        private static void swap(int[] indexes, int i, int j, int k, int m) {
            int jj = indexes[j];
            indexes[j] = indexes[k];
            indexes[k] = jj;
        }

        // Looks like swap() is its own inverse, since it swaps two elements, and swapping
        // them again restores the original order.
        private static void unswap(int[] indexes, int i, int j, int k, int m) {
            swap(indexes, i, j, k, m);
        }

        /**
         * The actual heuristic for generating successive solutions.
         */
        private void iterateOnce() {
            // Do nothing if we're as good as can be.
            if (reachedFixedPoint) return;

            int numVertices = context.getNumVertices();
            int[] bestReachableSolution = Arrays.copyOf(currentSolution, currentSolution.length);
            AtomicReference<Double> bestReachableCost =
                new AtomicReference<>(context.getPathCost(currentSolution));

            doActionForAllPairsOfEdges(
                bestReachableSolution,
                bestReachableCost,
                this::singleMoveInSearchSpace);

            // If we are in danger of getting stuck, do a heroic depth-2 search.
            if (context.getPathCost(currentSolution) == bestReachableCost.get()) {
                doActionForAllPairsOfEdges(
                    bestReachableSolution,
                    bestReachableCost,
                    (brs, brc, i, j, k, m) -> {
                        swap(currentSolution, i, j, k, m);
                        doActionForAllPairsOfEdges(brs, brc, this::singleMoveInSearchSpace);
                        unswap(currentSolution, i, j, k, m);
                    });
                // If still stuck, give up on further iteration.
                if (context.getPathCost(currentSolution) == bestReachableCost.get()) {
                    reachedFixedPoint = true;
                }
                // If looking one step ahead got us unstuck,
                // take the step from current state to the improved state we found.
                else System.arraycopy(bestReachableSolution, 0, currentSolution, 0, numVertices);
            }
            // If not stuck, take the step from current state to the improved state we found.
            else System.arraycopy(bestReachableSolution, 0, currentSolution, 0, numVertices);
        }
    }

    /**
     * A POJO representing a possible TSP solution, not necessarily a best solution.
     */
    public static final class TspSolution {
        private final int[] vertexVisitOrder;
        private final double cost;

        private TspSolution(TspContext context, int[] vertexVisitOrder) {
            this.vertexVisitOrder =
                Arrays.copyOf(vertexVisitOrder, vertexVisitOrder.length);
            this.cost = context.getPathCost(vertexVisitOrder);
        }

        public double getCost() {
            return this.cost;
        }

        public int[] getVertexVisitOrder() {
            return Arrays.copyOf(vertexVisitOrder, vertexVisitOrder.length);
        }
    }
}

package adm2e.tsp;

import adm2e.tsp.AnnealingSearch.HeuristicTspSolver.DecisionRule;
import adm2e.tsp.AnnealingSearch.HeuristicTspSolver.DecisionRule.Decision;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static adm2e.tsp.AnnealingSearch.HeuristicTspSolver.DecisionRule.Decision.ACCEPT;
import static adm2e.tsp.AnnealingSearch.HeuristicTspSolver.DecisionRule.Decision.REJECT;

/**
 * <p>A simple "annealing" TSP solver that always accepts improved solutions
 * and sometimes accepts bad solutions. Slightly worse bad solutions are
 * more acceptable than greatly worse solutions. The threshold for acceptance
 * also becomes more strict over time.</p>
 * <br />
 * <p>This solver works best for non-trivial problem spaces.</p>
 */
public class AnnealingSearch {

    private static final String TEMP_REDUCTION_FLAG = "r";
    private static final String STUCK_FLAG = "s";
    private static final String NUM_TRIALS_FLAG = "n";
    private static final int DEFAULT_NUM_TRIALS = 1;

    private static Options getOptions() {
        Options options = new Options();
        Option tempReductionThreshold = new Option(
            TEMP_REDUCTION_FLAG,
            "reduce-after",
            true, // option does have a value
            "reduce temperature after this many consecutively accepted state transitions");
        Option stuckCondition = new Option(
            STUCK_FLAG,
            "stop-after",
            true, // option does have a value
            "stop after this many consecutive iterations with no improvement in cost"
        );
        Option numRetries = new Option(
            NUM_TRIALS_FLAG,
            "num-trials",
            true,
            "number of trials to run"
        );
        options.addOption(tempReductionThreshold);
        options.addOption(stuckCondition);
        options.addOption(numRetries);
        return options;
    }

    private static void usage() {
        System.err.println(
            "Usage: java -cp 'out:lib' "
                + AnnealingSearch.class.getName()
                + " <path to input file> <number of retries>");
    }

    public static void main(String[] args) throws IOException, ParseException {
        CommandLine cli = new DefaultParser().parse(getOptions(), args, true);

        int tempReductionAfter;
        if (cli.hasOption(TEMP_REDUCTION_FLAG)) {
            tempReductionAfter = Integer.parseInt(cli.getOptionValue(TEMP_REDUCTION_FLAG));
        } else {
            tempReductionAfter = AnnealingRule.CONSECUTIVE_ACCEPTS_BEFORE_TEMP_REDUCED;
        }
        int stuckAfter;
        if (cli.hasOption(STUCK_FLAG)) {
            stuckAfter = Integer.parseInt(cli.getOptionValue("s"));
        } else {
            stuckAfter = AnnealingRule.MAX_CONSECUTIVE_SAME_CURRENT_COST;
        }
        int numTrials;
        if (cli.hasOption(NUM_TRIALS_FLAG)) {
            numTrials = Integer.parseInt(cli.getOptionValue(NUM_TRIALS_FLAG));
        } else {
            numTrials = DEFAULT_NUM_TRIALS;
        }

        Stream<String> lines = readRawInput(cli);

        // Initialize collections for parsing input.
        Map<String, Map<String, Double>> labelToLabelToDistance = new HashMap<>();
        Set<String> labelCollector = new HashSet<>();

        // Populate collections from input.
        lines.forEach(line ->
            processLine(line, labelCollector, labelToLabelToDistance));

        // Input was invalid.
        if (labelCollector.isEmpty()) return;

        // Reshape collected input.
        String[] labels = labelCollector.toArray(new String[0]);
        Arrays.sort(labels);
        double[][] distances = buildEdgeWeightMatrix(labels, labelToLabelToDistance);

        // Pick criterion for accepting candidate solutions.
        // The annealing rule explores the solution space more diligently by
        // occasionally picking suboptimal solutions to avoid getting stuck.
        // This solution rule is stateful, so the supplier creates distinct
        // instances on demand.
        Supplier<DecisionRule> decisionRuleSupplier = () -> new AnnealingRule(tempReductionAfter, stuckAfter);

        // Run the solver several times and pick the best candidate solution.
        TspSolution bestSolution = null;
        double bestCost = Double.MAX_VALUE;
        HeuristicTspSolver solver =
            HeuristicTspSolver.create(labels, distances, decisionRuleSupplier);
        for (int i = 0; i < numTrials; i++) {
            solver = solver.reinitializedCopy();
            TspSolution solution = solver.getFixedPointSolution();
            if (solution.getCost() < bestCost) {
                bestCost = solution.getCost();
                bestSolution = solution;
            }
        }
        System.out.println(bestSolution);
    }

    private static Stream<String> readRawInput(CommandLine cli) throws IOException {
        Path inputFile = Path.of(cli.getArgs()[0]);
        try {
            return Files.newBufferedReader(inputFile)
                .lines()
                .filter(s -> !s.isEmpty());
        } catch (IOException e) {
            new HelpFormatter().printHelp("java -cp 'out/*:libs/*' [JAVA_OPTS] " + AnnealingSearch.class.getName() + " ",
                getOptions());
            return Stream.empty();
        }
    }

    private static void processLine(String line,
                                    Set<String> labelCollector,
                                    Map<String, Map<String, Double>> distances) {
        // Tokenize line on whitespace.
        String[] tokens = line.split("\\s+");
        // Select TO and FROM so that FROM < TO lexicographically.
        // This avoids having both A -> B and B -> A in the parsed input.
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
     * Encapsulates the state transition logic for the solver. Always accepts
     * better solutions, but also sometimes accepts worse solutions, and as a
     * consequence gets itself unstuck sometimes. The probability of accepting
     * a worse solution starts high and gets reduced after an uninterrupted
     * sequence of acceptable state transitions. The rule decides that iteration
     * is stuck after an uninterrupted sequence of same-cost iterations.
     */
    private static final class AnnealingRule implements DecisionRule {
        // parameter defaults
        private static final double INITIAL_TEMPERATURE = 1.0;
        private static final int CONSECUTIVE_ACCEPTS_BEFORE_TEMP_REDUCED = 20;
        private static final double TEMP_REDUCTION_FACTOR = 0.5;
        private static final double TEMP_LOWER_THRESHOLD = Math.pow(2, -12);
        private static final int MAX_CONSECUTIVE_SAME_CURRENT_COST = 5;

        // parameters for this instance
        private final int consecutiveAcceptsBeforeTempReduced;
        private final int maxConsecutiveSameCurrentCost;

        // state
        private final Random random = ThreadLocalRandom.current();
        private double currentTemperature;
        private int consecutiveAccepts;
        private double lastSeenCurrentCost;
        private int consecutiveSameCurrentCost;

        private AnnealingRule(int consecutiveAcceptsBeforeTempReduced,
                              int maxConsecutiveSameCurrentCost) {
            this.consecutiveAcceptsBeforeTempReduced = consecutiveAcceptsBeforeTempReduced;
            this.maxConsecutiveSameCurrentCost = maxConsecutiveSameCurrentCost;
            this.currentTemperature = INITIAL_TEMPERATURE;
            this.consecutiveAccepts = 0;
            this.lastSeenCurrentCost = Double.MAX_VALUE;
            this.consecutiveSameCurrentCost = 0;
        }

        @Override
        public Decision apply(double currentCost, double nextCost) {
            // If nextCost <= currentCost, this value is at least equal to 1,
            // and the test should always pass. The test also passes if
            // the diff is close to zero -- solution readily moves
            // between equally good states.
            double criterion = Math.exp((currentCost - nextCost) / currentTemperature);
            if (criterion > random.nextDouble()) {
                consecutiveAccepts++;
                if (consecutiveAccepts > consecutiveAcceptsBeforeTempReduced) {
                    System.out.println("Halving temperature!");
                    consecutiveAccepts = 0;
                    currentTemperature = currentTemperature * TEMP_REDUCTION_FACTOR;
                }
                return ACCEPT;
            } else {
                return REJECT;
            }
        }

        @Override
        public boolean searchBudgetExceeded() {
            boolean exceeded = currentTemperature < TEMP_LOWER_THRESHOLD;
            if (exceeded) System.out.println("Temperature below threshold!");
            return exceeded;
        }

        @Override
        public boolean fixedPointDetected(double currentCost, double bestCost) {
            if (currentCost < lastSeenCurrentCost) {
                lastSeenCurrentCost = currentCost;
                consecutiveSameCurrentCost = 0;
                return false;
            }
            consecutiveSameCurrentCost++;
            boolean stuck = consecutiveSameCurrentCost >= maxConsecutiveSameCurrentCost;
            if (stuck) System.out.println("Got stuck!");
            return stuck;
        }
    }

    /**
     * Stores the search space for a TSP problem.
     */
    private static final class TspContext {
        private final String[] vertexLabels;
        private final double[][] vertexDistances;
        private final DecisionRule decisionRule;

        private TspContext(String[] vertexLabels,
                           double[][] vertexDistances,
                           DecisionRule decisionRule) {
            if (vertexLabels.length != vertexDistances.length
                || vertexDistances.length != vertexDistances[0].length) {
                throw new IllegalArgumentException(
                    "Given N vertices, the matrix of distances between them should be NxN.");
            }

            int numVertices = vertexLabels.length;
            this.vertexLabels = Arrays.copyOf(vertexLabels, numVertices);
            this.vertexDistances = copySquareSymmetricArray(vertexDistances);
            this.decisionRule = decisionRule;
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

        private static double[][] copySquareSymmetricArray(double[][] orig) {
            int size = orig.length;
            double[][] copy = new double[size][size];
            for (int i = 0; i < size; i++) {
                for (int j = i + 1; j < size; j++) {
                    copy[i][j] = orig[i][j];
                    copy[j][i] = copy[i][j];
                }
            }
            return copy;
        }

        /**
         * Edge weight between two vertices.
         */
        public double getDistance(int i, int j) {
            return vertexDistances[i][j];
        }

        public double[][] getVertexDistances() {
            return copySquareSymmetricArray(vertexDistances);
        }

        private DecisionRule getDecisionRule() {
            return decisionRule;
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
        private final Supplier<DecisionRule> decisionRuleSupplier;

        private HeuristicTspSolver(TspContext context,
                                   Supplier<DecisionRule> decisionRuleSupplier) {
            this.context = context;
            this.decisionRuleSupplier = decisionRuleSupplier;
            this.currentSolution = randomPermutation(context);
            this.reachedFixedPoint = false;
        }

        public static HeuristicTspSolver create(String[] labels,
                                                double[][] distances,
                                                Supplier<DecisionRule> decisionRuleSupplier) {
            TspContext context = new TspContext(labels, distances, decisionRuleSupplier.get());
            return new HeuristicTspSolver(context, decisionRuleSupplier);
        }

        /**
         * Create a copy of this solver with the same data and heuristic, but a
         * different initial solution that probably isn't stuck yet.
         */
        public HeuristicTspSolver reinitializedCopy() {
            TspContext copyContext = new TspContext(context.getVertexLabels(),
                context.getVertexDistances(),
                decisionRuleSupplier.get());
            return new HeuristicTspSolver(copyContext, decisionRuleSupplier);
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
            while (!reachedFixedPoint) {
                iterateOnce();
            }
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

        private void doActionForAllPairsOfEdges(int[] selectedReachableSolution,
                                                AtomicReference<Double> selectedReachableCost,
                                                SolverAction action) {
            int numVertices = context.getNumVertices();
            // First, edges <0, 1> and <2, 3>, <0, 1> and <4, 5>, ..., <2, 3> and <4, 5>, etc.
            for (int i = 0, j = 1;
                 j < numVertices - 2;
                 i += 2, j += 2) {

                for (int k = i + 2, m = j + 2;
                     m < numVertices;
                     k += 2, m += 2) {

                    action.perform(selectedReachableSolution, selectedReachableCost, i, j, k, m);
                }
            }
            // Then, edges <1, 2> and <3, 4>, <1, 2> and <5, 6>, ..., <3, 4> and <5, 6>, etc.
            for (int i = 1, j = 2;
                 j < numVertices - 2;
                 i += 2, j += 2) {

                for (int k = i + 2, m = j + 2;
                     m < numVertices;
                     k += 2, m += 2) {

                    action.perform(selectedReachableSolution, selectedReachableCost, i, j, k, m);
                }
            }
            // Finally, edge <0, LAST> and <1, 2>, <0, LAST> and <3, 4>, etc.
            for (int i = 0, j = numVertices - 1, k = 1, m = 2;
                 m < numVertices - 1;
                 k += 2, m += 2) {

                action.perform(selectedReachableSolution, selectedReachableCost, i, j, k, m);
            }
        }

        @FunctionalInterface
        interface DecisionRule {
            enum Decision { ACCEPT, REJECT }

            /**
             * Decides whether a proposed cost is better than the current cost.
             */
            Decision apply(double currentCost, double nextCost);

            /**
             * Hook for rules that terminate searching based on some internal state,
             * such as the number of times they've been called. The default is to run
             * forever.
             */
            default boolean searchBudgetExceeded() {
                return false;
            }

            /**
             * Hook for rules with arbitrary stopping conditions. The default is
             * to signal that a fixed point has been reached as soon as we reach
             * a local minimum.
             */
            default boolean fixedPointDetected(double currentCost, double bestCost) {
                return currentCost <= bestCost;
            }
        }

        // Just a simple functional interface to type-alias this ugly function signature.
        @FunctionalInterface
        private interface SolverAction {
            void perform(int[] selectedReachableSolution,
                         AtomicReference<Double> selectedReachableCost,
                         int i, int j, int k, int m);
        }

        // Default implementation of SolverAction: swap a pair of edges and invoke
        // the decision rule on the result.
        private void singleMoveInSearchSpace(int[] selectedReachableSolution,
                                             AtomicReference<Double> selectedReachableCost,
                                             int i, int j, int k, int m) {
            swap(currentSolution, i, j, k, m);
            double newCost = context.getPathCost(currentSolution);
            Decision decision =
                context.getDecisionRule().apply(selectedReachableCost.get(), newCost);
            if (decision == ACCEPT) {
                selectedReachableCost.set(newCost);
                System.arraycopy(currentSolution, 0, selectedReachableSolution, 0, context.getNumVertices());
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
            // Check if we should search at all.
            if (context.getDecisionRule().searchBudgetExceeded()) {
                reachedFixedPoint = true;
            }

            // Do nothing if we're as good as can be.
            if (reachedFixedPoint) return;

            int numVertices = context.getNumVertices();
            int[] selectedReachableSolution =
                Arrays.copyOf(currentSolution, currentSolution.length);
            AtomicReference<Double> selectedReachableCost =
                new AtomicReference<>(context.getPathCost(currentSolution));

            // Modifies selectedReachableSolution and selectedReachableCost.
            doActionForAllPairsOfEdges(
                selectedReachableSolution,
                selectedReachableCost,
                this::singleMoveInSearchSpace);

            double currentCost = context.getPathCost(currentSolution);
            double nextCost = selectedReachableCost.get();
            if (context.getDecisionRule().fixedPointDetected(currentCost, nextCost)) {
                reachedFixedPoint = true;
            }
            // If not stuck, take the step from current state to the improved state we found.
            else System.arraycopy(selectedReachableSolution, 0, currentSolution, 0, numVertices);
        }
    }

    /**
     * A POJO representing a possible TSP solution, not necessarily a best solution.
     */
    private static final class TspSolution {
        private final int[] vertexVisitOrder;
        private final double cost;
        private final TspContext context;

        private TspSolution(TspContext context, int[] vertexVisitOrder) {
            this.vertexVisitOrder =
                Arrays.copyOf(vertexVisitOrder, vertexVisitOrder.length);
            this.cost = context.getPathCost(vertexVisitOrder);
            this.context = context;
        }

        public double getCost() {
            return cost;
        }

        public int[] getVertexVisitOrder() {
            return Arrays.copyOf(canonicalizeVertexOrder(), vertexVisitOrder.length);
        }

        @Override
        public String toString() {
            return "Visit order: ["
                + Arrays.stream(canonicalizeVertexOrder())
                .mapToObj(context::getVertexLabel)
                .collect(Collectors.joining(" -> "))
                + "]. Total cost: "
                + getCost();
        }

        // A copy of the vertex order with vertex 0 in position 0.
        private int[] canonicalizeVertexOrder() {
            int minIndex = 0;
            int numVertices = vertexVisitOrder.length;
            for (int i = 1; i < numVertices; i++) {
                if (vertexVisitOrder[i] < vertexVisitOrder[minIndex]) {
                    minIndex = i;
                }
            }
            int[] canonical = new int[numVertices];
            System.arraycopy(vertexVisitOrder, minIndex, canonical, 0, numVertices - minIndex);
            System.arraycopy(vertexVisitOrder, 0, canonical, numVertices - minIndex, minIndex);
            return canonical;
        }
    }
}

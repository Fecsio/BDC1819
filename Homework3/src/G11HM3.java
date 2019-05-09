import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;



public class G11HM3 {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name, the desired number of clusters and the desired number of iterations on the command line");
        }

        ArrayList<Vector> P = readVectorsSeq(args[0]); // Reads the input points from the file into an ArrayList<Vector>

        ArrayList<Long> WP = new ArrayList<>(); // Declares ArrayList of weights
        WP.addAll(Collections.nCopies(P.size(), (long) 1)); // Sets all weights to one

        long start = System.currentTimeMillis();
        ArrayList<Vector> C = kmeansPP(P, WP, Integer.parseInt(args[1]), Integer.parseInt(args[2])); // Runs kmeansPP(P,WP,k,iter), obtaining a set of k centers C (k and iter are given from the command line)
        long end = System.currentTimeMillis();
        long execTime = end - start;

        System.out.println("Value of kmeansObj: " + kmeansObj(P,C)); // Runs kmeansObj(P,C) printing the returned value

        System.out.println("Execution time: " + (double)execTime/1000 + " seconds");

    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    public static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> WP, int k, int iter) {
        /**
         * Implementation of kmeansPP algorithm
         * @return a first set of k centroids to use as input in Lloyd's algorithm
         */

        Random rand = new Random();
        ArrayList<Vector> S = new ArrayList<>();
        // select and insert in S initial center with uniform probability and remove it from P
        int r = rand.nextInt(P.size());
        S.add(P.get(r));

        // initializes array of probabilities for center selection
        ArrayList<Double> probs = new ArrayList(Collections.nCopies(P.size(), 0));

        // initializes array of distances from every point to nearest center
        ArrayList<Double> min_distances = new ArrayList<>(Collections.nCopies(P.size(), Double.POSITIVE_INFINITY));

        for (int i = 1; i < k; i++) {

            // for each point p in P, check if the distance from p to the last found center is less than
            // the saved one (that will be the distance from p to the closest among all already selected centers);
            // if it's smaller, it overwrites the old one
            for (int j = 0; j < P.size(); j++) {

                double dist = Math.sqrt(Vectors.sqdist(P.get(j), S.get(S.size() - 1)));

                if (dist < min_distances.get(j)) {
                    min_distances.set(j, dist);
                }
            }


            // calculates denominator used to calculate the probability
            Double sum = Double.valueOf(0);
            for (int ii = 0; ii < WP.size(); ii++) {
                sum += WP.get(ii) * min_distances.get(ii);
            }

            // set probability of choosing each center
            for (int l = 0; l < probs.size(); l++) {
                double p = WP.get(l) * min_distances.get(l) / sum;
                probs.set(l, p);
            }

            // choose new center
            double p = Math.random();
            double cumulativeProbability = 0.0;
            for (int n = 0; n < probs.size(); n++) {
                cumulativeProbability += probs.get(n);
                if (p <= cumulativeProbability) {
                    Vector newC = P.get(n);
                    if (!S.contains(newC)) {
                        S.add(newC);
                        break;
                    }

                }
            }

        }

        return Lloyd(P, S, WP, k, iter);
    }

    public static ArrayList<Vector> Lloyd(ArrayList<Vector> P, ArrayList<Vector> S, ArrayList<Long> WP, int k, int iter) {
        /**
         * Implementation of LLoyd's algorithm
         * @return k centroids calculated after iter iterations of LLoyd's algorithm
         */

        for (int i = 0; i < iter; i++) {
            ArrayList<Vector> sumPoints = new ArrayList<>(); // sumPoints.get(j) will be sum_{p in Cj} p * w(p)
            long[] sumWeights = new long[k]; // sumWeights.get(j) will be sum_{p in Cj} w(p)

            for(int j = 0; j < S.size(); j++){
                sumPoints.add(Vectors.dense(new double[P.get(0).size()]));
            }

            for (int v = 0; v < P.size(); v++) { // for each point p in P, find distance from closest centroid in S
                 Vector p = P.get(v);
                if (!S.contains(p)) { // P - S
                    int bestClusterIndex = 0;
                    Double bestDist = Math.sqrt(Vectors.sqdist(p, S.get(0))); // calculates distance from first centroid in S
                    for (int s = 1; s < S.size(); s++) {
                        Vector newC = S.get(s);
                        Double tmpDist = Math.sqrt(Vectors.sqdist(p, newC)); // calculates distance from s-th centroid in S
                        if (tmpDist < bestDist) { // check if newly calculated distance is less than previous minimum distance
                            bestDist = tmpDist;
                            bestClusterIndex = s;
                        }
                    }

                    // here bestClusterIndex == index of centroid in S with minimum distance from p
                    long weight = WP.get(v);
                    BLAS.axpy(weight, p, sumPoints.get(bestClusterIndex));
                    sumWeights[bestClusterIndex] += weight;
                }
            }

            // here sumPoints.get(j) = sum_{p in Cj} p * w(p) and
            // sumWeight[j] = sum_{p in Cj} w(p)

            for (int j = 0; j < k; j++) { // calculating new k-centroids

                if (sumWeights[j] > 0) {
                    BLAS.scal(Double.valueOf(1) / sumWeights[j], sumPoints.get(j)); // using sumWeights and sumPoints, calculates new j-th centroid
                } else {
                    System.out.println("WARNING: empty cluster found");
                }

                S.set(j, sumPoints.get(j)); // saves new j-th centroid in S
            }

        }

        return S;
    }


    public static double kmeansObj(ArrayList<Vector> P, ArrayList<Vector> C) {

        double phi = 0;

        for(Vector p : P){

            double bestdist = Math.sqrt(Vectors.sqdist(p, C.get(0)));

            for (int c = 1; c<C.size(); c++){
                double tempDist = Math.sqrt(Vectors.sqdist(p, C.get(c)));

                if(tempDist < bestdist){
                    bestdist = tempDist;
                }
            }

            phi += bestdist;
        }

        return phi/P.size();
    }
}

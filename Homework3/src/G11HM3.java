import breeze.stats.distributions.Rand;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import  org.apache.spark.mllib.linalg.BLAS;
import org.json4s.DefaultWriters;

import java.io.*;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;


public class G11HM3 {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        ArrayList<Vector> P = readVectorsSeq(args[0]);
        ArrayList<Long> WP = new ArrayList<>();
        WP.addAll(Collections.nCopies(10000, (long)1));

        ArrayList<Vector> C = kmeansPP(P, WP, 2, 0);
        // r.forEach(v -> System.out.println(v));


    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
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

        Random rand = new Random();
        ArrayList<Vector> S = new ArrayList<>();
        // select and insert in S initial center with uniform probability and remove it from P
        int r = rand.nextInt(P.size());
        S.add(P.remove(r));
        WP.remove(r);

        // initializes array of probabilities for center selection
        ArrayList<Double> probs = new ArrayList(Collections.nCopies(P.size(), 0));

        // initializes array of distances from every point to nearest center
        ArrayList<Double> min_distances = new ArrayList<>(Collections.nCopies(P.size(), Double.POSITIVE_INFINITY));

        for (int i = 2; i <= k; i++) {

            // for each point p in (old)P-S, check if the distance from p to the last found center is less than
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
            for (int ii = 0; i < P.size(); ii++) {
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
                    P.remove(n);
                    WP.remove(n);
                    S.add(newC);
                }
            }

        }

        return Lloyd(S, iter);
    }

    // We pass only P, assuming S elements have already been removed in kmeanspp
    public static ArrayList<Vector> Partition(ArrayList<Vector> P){
        
    }

    public static ArrayList<Vector> Lloyd(ArrayList<Vector> S, int iter) {

    }







    }

}

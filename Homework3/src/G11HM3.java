import breeze.stats.distributions.Rand;
import org.apache.commons.math3.ml.distance.ManhattanDistance;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.json4s.DefaultWriters;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import javax.mail.Part;
import javax.swing.plaf.synth.SynthTextAreaUI;
import java.io.*;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static org.apache.spark.mllib.linalg.BLAS.*;


public class G11HM3 {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        ArrayList<Vector> P = readVectorsSeq(args[0]);
        ArrayList<Long> WP = new ArrayList<>();
        WP.addAll(Collections.nCopies(P.size(), (long) 1));

        long start = System.currentTimeMillis();
        ArrayList<Vector> C = kmeansPP(P, WP, 15, 100);
        long end = System.currentTimeMillis();
        long execTime = end - start;

        System.out.println(kmeansObj(P,C));
        System.out.println("tempo: " + execTime/1000 + " seconds");


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

            // for each point p in P-S, check if the distance from p to the last found center is less than
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

    public static HashMap<Vector, ArrayList<Vector>> Partition(ArrayList<Vector> P, ArrayList<Vector> S) {

        HashMap<Vector, ArrayList<Vector>> C = new HashMap<>();
        for (Vector p : P) {
            if (!S.contains(p)) {
                Vector bestCluster = S.get(0);
                Double bestDist = Math.sqrt(Vectors.sqdist(p, bestCluster));
                for (Vector s : S) {
                    if (s != bestCluster) {
                        Double tmpDist = Math.sqrt(Vectors.sqdist(p, s));
                        if (tmpDist < bestDist) {
                            bestDist = tmpDist;
                            bestCluster = s;
                        }
                    }
                }
                C.computeIfAbsent(bestCluster, k -> new ArrayList<>()).add(p);
            }
        }

        for (Vector s : S) {
            C.computeIfAbsent(s, k -> new ArrayList<>());
        }

        return C;
    }

    public static ArrayList<Vector> Lloyd(ArrayList<Vector> P, ArrayList<Vector> S, ArrayList<Long> WP, int k, int iter) {

        Double phi = kmeansObj(P,S);

        boolean stop = false;
        ArrayList<Vector> centroids = new ArrayList<>();

        int ii = 0;
        for (int i = 0; i < iter && !stop; i++) {
            HashMap<Vector, ArrayList<Vector>> C = Partition(P, S); //key = centers, values = points in cluster
            centroids = new ArrayList<>(C.keySet()); //centers

            for (int j = 0; j < k; j++) {
                Vector tmp = centroids.get(j); //j-th center
                ArrayList<Vector> clusterJPoints = C.get(tmp); // points in cluster with center = j-th cluster
                int clusterJSize = C.get(tmp).size(); //size of cluster with center = j-th cluster

                if (clusterJSize > 0) {
                    clusterJPoints.forEach(vector -> scal(WP.get(P.indexOf(vector)), vector)); // w(p) * p
                    Vector sum = new DenseVector(new double[clusterJPoints.get(0).size()]);
                    BLAS.copy(clusterJPoints.get(0), sum);
                    for (int v = 1; v < clusterJPoints.size(); v++) { // sum for all p in Cj
                        axpy(1, clusterJPoints.get(v), sum);
                    }

                    scal(Double.valueOf(1) / clusterJSize, sum); //final calculation of centroid of cluster Cj

                    centroids.set(j, sum);
                    ArrayList<Vector> tempPoints = C.remove(tmp);
                    C.put(sum, tempPoints);
                } else {
                    System.out.println("WARNING: empty cluster");

                }
            }

            Double phikm = kmeansObj(P, centroids);

           // if(phikm < phi){
            phi = phikm;
            S.clear();
            S.addAll(centroids);



            int count = 0;

            /*try (PrintWriter p = new PrintWriter(new BufferedWriter(new FileWriter("Homework3/output.txt", true)))) {
                p.println("\n Iteraction: " + ii + "\n");
                p.println("\n Phi_k_means = " + phikm + "\n");
                if (phikm >= phi) p.println("\n STOP \n");
                if (S.equals(centroids)) p.println("\n STOP2 \n");






                Integer[] numArray = new Integer[k];

                for (Vector c : centroids) {
                    numArray[count] = C.get(c).size();
                    count++;
                    p.println("Size of cluster " + count + ": " + C.get(c).size());
                }

                double median;
                Arrays.sort(numArray);
                if (numArray.length % 2 == 0)
                    median = ((double) numArray[numArray.length / 2] + (double) numArray[numArray.length / 2 - 1]) / 2;
                else
                    median = (double) numArray[numArray.length / 2];

                p.println("Median: " + median);

            } catch (IOException ex) {
                System.out.println(ex.toString());
            }


            /*} else {
                System.out.println("\n Phi_k_means = " + phikm + "\n");
                stop = true;
            }*/


            /*if (i == iter - 1 || stop) {
                 count = 0;
                try (PrintWriter p = new PrintWriter(new BufferedWriter(new FileWriter("Homework3/output.txt", true)))) {
                    for (Vector v : C.keySet()) {
                        count++;
                        p.println("\nCLUSTER:" + count + "\n");
                        C.get(v).forEach(vv -> p.println(vv));
                    }
                } catch (IOException ex) {
                    System.out.println(ex.toString());
                }
            }*/
        }

        //System.out.println("# of iterations:" + ii);
        //System.out.println(kmeansObj(P,S));
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

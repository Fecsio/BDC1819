import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.lang.String;
import scala.Tuple2;


public class G11HM2 {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }
        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("HW2")
                .setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Parameter k given by cmd line
        int k = Integer.parseInt(args[1]);

        // 1 - Reads the collection of documents into an RDD docs and subdivides the into K parts
        JavaRDD<String> docs = sc.textFile(args[0], k).cache();
        docs.count();

        /* 2.1 - code for the Improved Word count 1 algorithm described in class the using
         reduceByKey method to compute the final counts */

        long wc1Start = System.currentTimeMillis();
        JavaPairRDD<String, Long> wordcountpairs = docs
                // Map phase: Creating the pairs (w,count) for each document
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Reduce phase: summing the occurrences of each word in every document
                .reduceByKey((x,y) -> x+y);

        long wcp = wordcountpairs.count();


        long wc1End = System.currentTimeMillis();

        /* 2.2.1 - code fore a variant of the algorithm presented in class where random keys take K possible values,
        where K is the value given in input*/

        long wc2Start = System.currentTimeMillis();
        JavaPairRDD<String, Long> wordcountpairs2 = docs
                // Map - 1: Creating the pairs (w,count) for each document and then assign to every pair
                // a random key x between (0-k)
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                }).groupBy(x -> new Long((int)(Math.random()*k)))
                // Reduce - 1 sum the number of occurrences of a specific w in the partition with key = x
                .flatMapToPair(x -> {
                    Iterator<Tuple2<String, Long>> iter = x._2().iterator();
                    HashMap<String, Long> count = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    while(iter.hasNext()) {
                        Tuple2 t = (Tuple2)iter.next();
                        String key = (String) t._1();
                        Long value = (Long) t._2();
                        count.put(key, value + count.getOrDefault(key,0L));
                    }
                    for(Map.Entry<String, Long> e : count.entrySet()){
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // map - 2: Identity
                // reduce - 2: Sum the occurrences of every word.
                .reduceByKey((x,y) -> x+y);

        long wcp2 = wordcountpairs2.count();


        long wc2End = System.currentTimeMillis();

        // 2.2.2 code for a variant that does not explicitly assign random keys but exploits the subdivision of docs
        // into K parts in combination with mapPartitionToPair to access each partition separately

        long wc3Start = System.currentTimeMillis();
        JavaPairRDD<String, Long> wordcountpairs3 = docs
                // Map - 1: Creating the pairs (w,count) for each document and then use the k partitions of
                // docs instead of assigning a random key to each pair
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Reduce - 1 Using the k partitions to create the pairs (w,c(x,w)) where x is one of the partitions
                .mapPartitionsToPair(x -> {
                    Iterator<Tuple2<String, Long>> iter = x;
                    HashMap<String, Long> count = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    while(iter.hasNext()) {
                        Tuple2 t = (Tuple2)iter.next();
                        String key = (String) t._1();
                        Long value = (Long) t._2();
                        count.put(key, value + count.getOrDefault(key,0L));
                    }
                    for(Map.Entry<String, Long> e : count.entrySet()){
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Map - 2 Identity
                // Reduce - 2 Sum the occurrences of every word
                .reduceByKey((x,y) -> x+y);

        long wcp3 = wordcountpairs3.count();

        long wc3End = System.currentTimeMillis();


        long countStart = System.currentTimeMillis();

        // 3 - Counting the average words's length, we take an RDD computed before of composed by distinct words
        Double sum = wordcountpairs
                // Map phase: for each word create a double value that represents the length of the word
                .map(x -> (double)x._1().length())
                // Reduce phase: Sum all the values obtained in the map phase to get the sum of the length of every word
                .reduce((x,y) -> x+y);
        // Computing the average by dividing the sum obtained before by the number of words obtained during one of the
        // word count computation done before
        double avg = sum/wcp;

        long countEnd = System.currentTimeMillis();


        // Printing the time elapsed for each implementation of word count
        System.out.println("average length of distinct words: " + avg);
        System.out.println("Elapsed time for word count 1: " + (wc1End - wc1Start) + " ms");
        System.out.println("Elapsed time for word count 2: " + (wc2End - wc2Start) + " ms");
        System.out.println("Elapsed time for word count 3: " + (wc3End - wc3Start) + " ms");
        System.out.println("Elapsed time for calculating the avarage length of a word: " + (countEnd - countStart) + " ms");
        //System.in.read();



    }


}

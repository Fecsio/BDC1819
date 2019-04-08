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
                // Map phase
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
                // Reduce phase
                .reduceByKey((x,y) -> x+y);
        long wc1End = System.currentTimeMillis();

        /* 2.2.1 - code fore a variant of the algorithm presented in class where random keys take K possible values,
        where K is the value given in input*/

        long wc2Start = System.currentTimeMillis();
        JavaPairRDD<String, Long> wordcountpairs2 = docs
                // Map - 1
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
                // Reduce - 1
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
                // reduce - 2
                .reduceByKey((x,y) -> x+y);

        long wc2End = System.currentTimeMillis();

        // 2.2.2 code for a variant that does not explicitly assign random keys but exploits the subdivision of docs
        // into K parts in combination with mapPartitionToPair to access each partition separately
        long wc3Start = System.currentTimeMillis();
        JavaPairRDD<String, Long> wordcountpairs3 = docs
                // Map - 1
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
                    return pairs.iterator();})
                // Reduce - 1
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
                    return pairs.iterator();})
                // Map - 2 Identity
                // Reduce - 2 
                .reduceByKey((x,y) -> x+y);

        long wc3End = System.currentTimeMillis();

        /*// Printing the list of counted words
        wordcountpairs3.foreach(data -> {
            System.out.println("Word: " + data._1() + "---- Count: "+ data._2());
        });*/


        // Printing the time elapsed for each implementation of word count
        System.out.println("Elapsed time for word count 1: " + (wc1End - wc1Start) + " ms");
        System.out.println("Elapsed time for word count 2: " + (wc2End - wc2Start) + " ms");
        System.out.println("Elapsed time for word count 3: " + (wc3End - wc3Start) + " ms");
        //System.in.read();

    }


}

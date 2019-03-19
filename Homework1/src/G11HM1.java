import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;


public class G11HM1 {

    // Comparator class necessary to use the max method, implements the compare method which returns -1 if the first
    // argument is higher and 1 if the second argument is higher otherwise returns 0
    public static class Dcomparator implements Serializable, Comparator<Double> {

        public int compare(Double a, Double b) {
            if (a < b) return -1;
            else if (a > b) return 1;
            return 0;
        }

    }

    public static void main(String[] args) throws FileNotFoundException {
        //System.setProperty("hadoop.home.dir", "C:/winutils"); // istruction to make spark worn on my machine
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        // Read a list of numbers from the program options
        ArrayList<Double> lNumbers = new ArrayList<>();
        Scanner s =  new Scanner(new File(args[0]));
        while (s.hasNext()){
            lNumbers.add(Double.parseDouble(s.next()));
        }
        s.close();

        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("Preliminaries");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a parallel collection
        JavaRDD<Double> dNumbers = sc.parallelize(lNumbers);

        double sumOfSquares = dNumbers.map((x) -> x*x).reduce((x, y) -> x + y);
        System.out.println("The sum of squares is " + sumOfSquares);

        // Max Calculated with map and reduce methods
        double max = dNumbers.reduce((x,y) -> {
            if (x > y) return x;
            else return y;
        });
        System.out.println("the maximum value found with the reduce method is: " + max);

        // Max calculated with RDD's max method
        double maxRDD = dNumbers.max(new Dcomparator());
        System.out.println("the maximum value found with the max method from RDD interface is: " + maxRDD);

        // Creaiting an RDD containing values normalized in [0,1]
        //JavaRDD<Double> dNormalized = sc.parallelize(lNumbers).map(x -> x/maxRDD); //alternative way
        JavaRDD<Double> dNormalized = dNumbers.map(x -> x/maxRDD);

        // Now we compute some statistics on dNormalized using some methods from the RDD interface

        //Prints the smallest 3 values in the list using RDD's takeOrdered method
        List<Double> smallest = dNormalized.takeOrdered(3);
        System.out.println("List of the 3 smallest values in dNormalized: " + smallest );

        //Prints the sum of the values higher then 0.5
        double upperSum = dNormalized.filter(x -> x > 0.5).reduce((x,y) -> x+y);
        System.out.println("Sum of values higher then 0.5: " + upperSum );

        //Prints the geometric mean of the values
        double geomMean = Math.pow(dNormalized.reduce((x,y) -> x * y), 1.0/dNormalized.count());
        System.out.println("The geometric mean calculated on the dataset is: " + geomMean);
    }
}
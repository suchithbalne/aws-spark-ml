package com.prediction;

import org.apache.spark.sql.functions;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.io.IOException;

/*
 * Java program to read the training and testing data from AWS S3 and train a Random Forest model to predict the quality of the wine.
 * @author Suchith Sameeri Balne
 */

public class WineQualityPrediction {
    public static void main(String[] args) {
        String trainingDataPath = "";
        String testDataPath = "";
        String outputPath = "";
        if (args.length > 3) {
            System.err.println("Usage: WineQualityPrediction <path-to-training-data> <path-to-test-data>");
            System.exit(1);
        } else if(args.length ==3){
            trainingDataPath = args[0];
            testDataPath = args[1];
            outputPath = args[2] + "model";
        } else{
            trainingDataPath = "s3://njit-cs643-pa2/TrainingDataset.csv";
            testDataPath = "s3://njit-cs643-pa2/ValidationDataset.csv";
            outputPath = "s3://njit-cs643-pa2/model";
        
        }
        
        SparkSession spark = SparkSession.builder()
        .appName("WineQualityPrediction")
        .config("spark.master", "local")
        .getOrCreate();
        
        // Create a JavaSparkContext object from the SparkSession object.
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        spark.sparkContext().setLogLevel("ERROR");

        
        // Load and parse the training data file from AWS S3, converting it to a DataFrame.
        Dataset<Row> trainingData = spark.read().format("csv")
        .option("header", true)
        .option("quote", "\"") // handle escaped quotes
        .option("delimiter", ";")
        .load(trainingDataPath);
        //Dataset<Row> trainingData = spark.read().format("csv").option("header", "true").option("inferSchema", "true").option("delimiter", ";").load("s3://njit-cs643-pa2/TrainingDataset.csv");
        
        
        
        // Load and parse the testing data file from AWS S3, converting it to a DataFrame.
        //Dataset<Row> testData = spark.read().format("csv").option("header", "true").option("inferSchema", "true").option("delimiter", ";").load("s3://njit-cs643-pa2/ValidationDataset.csv");
        
        Dataset<Row> testData = spark.read().format("csv")
        .option("header", true)
        .option("quote", "\"") // handle escaped quotes
        .option("delimiter", ";")
        .load(testDataPath);

        // Prepare training data from a list of (label, features) tuples.
        String[] inputColumns = {"fixed acidity", "volatile acidity", "citric acid", "chlorides", "total sulfur dioxide", "density", "sulphates", "alcohol"};
        
        // Cast all columns to double 
        for (String col : inputColumns) {
            trainingData = trainingData.withColumn(col, trainingData.col(col).cast("Double")); 
        }
        
        // Cast all input columns to double
        for (String col : inputColumns) {
            testData = testData.withColumn(col, testData.col(col).cast("Double"));
        }
        
        // Cast quality column to double
        trainingData = trainingData.withColumn("quality", trainingData.col("quality").cast("Double"));
        testData = testData.withColumn("quality", testData.col("quality").cast("Double"));
        
        trainingData = trainingData.withColumn("label", functions.when(trainingData.col("quality").geq(7), 1.0).otherwise(0.0));
        testData = testData.withColumn("label", functions.when(testData.col("quality").geq(7), 1.0).otherwise(0.0));
        
        // Vector Assemler
        VectorAssembler assembler = new VectorAssembler()
        .setInputCols(inputColumns)
        .setOutputCol("features");
        
        // Create a RandomForestClassifier.
        RandomForestClassifier rf = new RandomForestClassifier()
        .setLabelCol("quality")
        .setFeaturesCol("features")
        .setNumTrees(150)
        .setMaxBins(8)
        .setMaxDepth(15)
        .setSeed(150)
        .setImpurity("gini");
        
        // Configure an ML pipeline, which consists of assembler and random forest classifier.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rf});
        
        // Train model. This also runs the assembler.
        PipelineModel model = pipeline.fit(trainingData);
        
        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);
        
        // Selecting rows to display.
        predictions.select("prediction", "quality", "features").show(5);
        
        // Selecting (prediction, true label).
        MulticlassClassificationEvaluator mcevaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality")
        .setPredictionCol("prediction");
        
        double accuracy = mcevaluator.setMetricName("accuracy").evaluate(predictions);
        double f1Score = mcevaluator.setMetricName("f1").evaluate(predictions);
        
        
        System.out.println("Test Accuracy = " + accuracy);
        System.out.println("Test F1 Score = " + f1Score);
        
        try {
            model.write().overwrite().save(outputPath);
        } catch (IOException e) {
            System.err.println("Failed to save the model: " + e.getMessage());
        }
    }
}


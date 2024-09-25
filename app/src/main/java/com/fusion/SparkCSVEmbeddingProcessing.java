package com.fusion;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.output.Response;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.functions;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SparkCSVEmbeddingProcessing {

    public static void main(String[] args) throws IOException {

        // Step 1: Set up Spark Session and Context
        SparkConf conf = new SparkConf().setAppName("CSV Embedding Processing").setMaster("local[*]");
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

        // Path to the input CSV file with text
        String inputCsvPath = SparkCSVEmbeddingProcessing.class.getClassLoader().getResource("data/ag_news_1000.csv").getPath();

        String outputCsvPath = "/path/to/your/output_with_embeddings.csv";

        // Step 2: Read the CSV file using Spark
        Dataset<Row> csvData = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(inputCsvPath);
        csvData.show(10);

        // Assuming the CSV has a column named "text"
        JavaRDD<DocumentData> documentDataRDD = csvData.select("text").javaRDD().map(row -> {
            String text = row.getString(0);  // Extract text from the CSV row
            return new DocumentData(text);   // Return a new DocumentData object
        });

        // Step 3: Generate embeddings for the extracted text using LangChain4J
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        JavaRDD<DocumentDataWithEmbedding> documentDataWithEmbeddingRDD = documentDataRDD.map(document -> {
            Response<Embedding> response = embeddingModel.embed(document.getText());
            float[] embeddings = response.content().vector();
            return new DocumentDataWithEmbedding(document.getText(), embeddings);
        });

        // Step 4: Write the output text and embeddings to a CSV
        List<DocumentDataWithEmbedding> resultList = documentDataWithEmbeddingRDD.collect();
        writeOutputToCsv(resultList, outputCsvPath);

        // Stop the Spark context
        sc.stop();
    }

    // Helper method to write output text and embeddings to CSV
    private static void writeOutputToCsv(List<DocumentDataWithEmbedding> data, String outputCsvPath) throws IOException {
        FileWriter csvWriter = new FileWriter(outputCsvPath);
        csvWriter.append("Text,Embeddings\n");
        for (DocumentDataWithEmbedding row : data) {
            csvWriter.append("\"").append(row.getText().replace("\"", "\"\"")).append("\",");
            csvWriter.append("\"").append(arrayToString(row.getEmbeddings())).append("\"\n");
        }
        csvWriter.flush();
        csvWriter.close();
    }

    // Helper method to convert embeddings array to a string representation
    private static String arrayToString(float[] array) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i]);
            if (i < array.length - 1) sb.append(",");
        }
        return sb.toString();
    }

    // Class to hold document data (text only)
    public static class DocumentData {
        private String text;

        public DocumentData(String text) {
            this.text = text;
        }

        public String getText() {
            return text;
        }
    }

    // Class to hold document data with embeddings
    public static class DocumentDataWithEmbedding extends DocumentData {
        private float[] embeddings;

        public DocumentDataWithEmbedding(String text, float[] embeddings) {
            super(text);
            this.embeddings = embeddings;
        }

        public float[] getEmbeddings() {
            return embeddings;
        }
    }
}
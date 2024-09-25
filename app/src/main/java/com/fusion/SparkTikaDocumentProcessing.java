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
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.sax.BodyContentHandler;
import org.xml.sax.ContentHandler;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SparkTikaDocumentProcessing {

    public static void main(String[] args) {

        // Step 1: Set up Spark Session and Context
        SparkConf conf = new SparkConf().setAppName("Document Processing with Tika and Spark").setMaster("local[*]");
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

        // Directory containing the documents (you can change this to your directory)
        String inputDirectory = "/path/to/your/documents";

        // Step 2: Parallelize the file paths in the directory and subdirectories
        List<File> files = listFilesInDirectory(new File(inputDirectory));
        JavaRDD<File> filesRDD = sc.parallelize(files);

        // Step 3: Process each file using Tika to extract the first two pages
        JavaRDD<DocumentData> documentDataRDD = filesRDD.map(file -> {
            String extractedText = extractTextFromFirstTwoPages(file);
            return new DocumentData(file.getName(), file.getAbsolutePath(), extractedText);
        });

        // Step 4: Generate embeddings for the extracted text using LangChain4J
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        JavaRDD<DocumentDataWithEmbedding> documentDataWithEmbeddingRDD = documentDataRDD.map(document -> {
            Response<Embedding> response = embeddingModel.embed(document.getExtractedText());
            float[] embeddings = response.content().vector();
            return new DocumentDataWithEmbedding(document.getFileName(), document.getFilePath(), document.getExtractedText(), embeddings);
        });

        // Step 5: Convert the result to a DataFrame and write it to a CSV
        Dataset<Row> documentDataFrame = spark.createDataFrame(documentDataWithEmbeddingRDD, DocumentDataWithEmbedding.class);
        documentDataFrame.write().format("csv").save("/path/to/save/output.csv");

        // Stop the Spark context
        sc.stop();
    }

    // Method to list all files in a directory, including subdirectories
    private static List<File> listFilesInDirectory(File dir) {
        List<File> resultList = new ArrayList<>();
        File[] fileList = dir.listFiles();
        if (fileList != null) {
            for (File file : fileList) {
                if (file.isFile()) {
                    resultList.add(file);
                } else if (file.isDirectory()) {
                    resultList.addAll(listFilesInDirectory(file));
                }
            }
        }
        return resultList;
    }

    // Method to extract text from the first two pages of a document using Apache Tika
    private static String extractTextFromFirstTwoPages(File file) {
        try (FileInputStream inputStream = new FileInputStream(file)) {
            ContentHandler handler = new BodyContentHandler(100000); // Character limit
            Metadata metadata = new Metadata();
            AutoDetectParser parser = new AutoDetectParser();
            parser.parse(inputStream, handler, metadata);
            // Assume the document is split into pages by newlines; only get first two pages
            String[] pages = handler.toString().split("\n\n"); // You may need to adjust this splitting logic based on actual format
            StringBuilder firstTwoPages = new StringBuilder();
            for (int i = 0; i < Math.min(2, pages.length); i++) {
                firstTwoPages.append(pages[i]).append("\n");
            }
            return firstTwoPages.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    // Class to hold document data (file name, path, and extracted text)
    public static class DocumentData {
        private String fileName;
        private String filePath;
        private String extractedText;

        public DocumentData(String fileName, String filePath, String extractedText) {
            this.fileName = fileName;
            this.filePath = filePath;
            this.extractedText = extractedText;
        }

        public String getFileName() {
            return fileName;
        }

        public String getFilePath() {
            return filePath;
        }

        public String getExtractedText() {
            return extractedText;
        }
    }

    // Class to hold document data with embeddings
    public static class DocumentDataWithEmbedding extends DocumentData {
        private float[] embeddings;

        public DocumentDataWithEmbedding(String fileName, String filePath, String extractedText, float[] embeddings) {
            super(fileName, filePath, extractedText);
            this.embeddings = embeddings;
        }

        public float[] getEmbeddings() {
            return embeddings;
        }
    }
}

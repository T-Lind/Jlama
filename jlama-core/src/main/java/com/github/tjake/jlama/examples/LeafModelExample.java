/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.examples;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;

import java.io.File;

/**
 * Example demonstrating how to use the LEAF embedding model (MongoDB/mdbr-leaf-ir) with Jlama.
 * 
 * The LEAF model is a compact 6-layer, 384-dimensional BERT-based embedding model
 * optimized for information retrieval tasks.
 * 
 * Usage:
 *   java -cp ... com.github.tjake.jlama.examples.LeafModelExample [model-directory]
 * 
 * If model-directory is not provided, it will attempt to download the model from HuggingFace.
 */
public class LeafModelExample {
    
    public static void main(String[] args) throws Exception {
        String modelName = "MongoDB/mdbr-leaf-ir";
        String workingDirectory = args.length > 0 ? args[0] : "./models";
        
        System.out.println("=== LEAF Model Integration Test ===");
        System.out.println("Model: " + modelName);
        System.out.println("Working directory: " + workingDirectory);
        System.out.println();
        
        // Download the LEAF model or use existing if already downloaded
        System.out.println("Downloading or locating LEAF model...");
        File localModelPath = SafeTensorSupport.maybeDownloadModel(workingDirectory, modelName);
        System.out.println("Model located at: " + localModelPath.getAbsolutePath());
        System.out.println();
        
        // Load as embedding model
        System.out.println("Loading model...");
        AbstractModel model = ModelSupport.loadEmbeddingModel(localModelPath, DType.F32, DType.F32);
        System.out.println("Model loaded successfully!");
        System.out.println("Embedding dimensions: " + model.getConfig().embeddingLength);
        System.out.println("Expected: 384");
        System.out.println();
        
        if (model.getConfig().embeddingLength != 384) {
            System.err.println("WARNING: Expected 384 dimensions but got " + model.getConfig().embeddingLength);
        }
        
        // Test embedding generation with AVG pooling (LEAF doesn't have a pooler layer)
        System.out.println("=== Testing Embedding Generation ===");
        String query1 = "What is artificial intelligence?";
        // LEAF model doesn't have a pooler layer, so we use AVG pooling
        Generator.PoolingType poolingType = Generator.PoolingType.AVG;
        float[] embedding1 = model.embed(query1, poolingType);
        System.out.println("Query: " + query1);
        System.out.println("Using AVG pooling (LEAF model doesn't have pooler layer)");
        System.out.println("Embedding dimension: " + embedding1.length);
        
        // Verify embedding values are finite and not all zeros
        boolean hasNonZero = false;
        boolean allFinite = true;
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        for (float v : embedding1) {
            if (v != 0.0f) hasNonZero = true;
            if (!Float.isFinite(v)) allFinite = false;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        System.out.println("Has non-zero values: " + hasNonZero);
        System.out.println("All values finite: " + allFinite);
        System.out.println("Value range: [" + min + ", " + max + "]");
        System.out.println();
        
        // Test similarity between related texts
        System.out.println("=== Testing Semantic Similarity ===");
        String query2 = "Define artificial intelligence";
        String query3 = "What is the weather today?";
        float[] embedding2 = model.embed(query2, poolingType);
        float[] embedding3 = model.embed(query3, poolingType);
        
        float similarity12 = VectorMath.cosineSimilarity(embedding1, embedding2);
        float similarity13 = VectorMath.cosineSimilarity(embedding1, embedding3);
        
        System.out.println("Query 1: " + query1);
        System.out.println("Query 2: " + query2);
        System.out.println("Query 3: " + query3);
        System.out.println();
        System.out.println("Similarity between Query 1 and Query 2: " + String.format("%.4f", similarity12));
        System.out.println("Similarity between Query 1 and Query 3: " + String.format("%.4f", similarity13));
        System.out.println();
        
        if (similarity12 > similarity13) {
            System.out.println("✓ Related queries have higher similarity (as expected)");
        } else {
            System.out.println("⚠ WARNING: Related queries should have higher similarity");
        }
        System.out.println();
        
        // Test with information retrieval examples
        System.out.println("=== Information Retrieval Test ===");
        String base = "MongoDB is a NoSQL database";
        String[] examples = new String[] {
            "MongoDB stores data in documents",
            "PostgreSQL is a relational database",
            "The cat sat on the mat",
            "NoSQL databases are non-relational",
            "MongoDB uses BSON format"
        };
        
        float[] baseEmbedding = model.embed(base, poolingType);
        float maxSimilarity = 0.0f;
        String bestMatch = "";
        int bestIndex = -1;
        
        System.out.println("Base query: " + base);
        System.out.println();
        for (int i = 0; i < examples.length; i++) {
            float[] exampleEmbedding = model.embed(examples[i], poolingType);
            float similarity = VectorMath.cosineSimilarity(baseEmbedding, exampleEmbedding);
            System.out.println(String.format("  [%d] Similarity: %.4f - %s", i, similarity, examples[i]));
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestMatch = examples[i];
                bestIndex = i;
            }
        }
        System.out.println();
        System.out.println("Best match: [" + bestIndex + "] " + bestMatch + " (similarity: " + String.format("%.4f", maxSimilarity) + ")");
        
        // The best match should be one of the MongoDB-related examples (indices 0, 3, or 4)
        if (bestIndex == 0 || bestIndex == 3 || bestIndex == 4) {
            System.out.println("✓ Best match is MongoDB-related (as expected)");
        } else {
            System.out.println("⚠ WARNING: Best match should be MongoDB-related");
        }
        System.out.println();
        
        // Performance test
        System.out.println("=== Performance Test ===");
        long start = System.currentTimeMillis();
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            model.embed(query1, poolingType);
        }
        long elapsed = System.currentTimeMillis() - start;
        double avgTime = (double) elapsed / iterations;
        System.out.println("Generated " + iterations + " embeddings in " + elapsed + "ms");
        System.out.println("Average time per embedding: " + String.format("%.2f", avgTime) + "ms");
        System.out.println();
        
        System.out.println("=== Test Complete ===");
    }
}


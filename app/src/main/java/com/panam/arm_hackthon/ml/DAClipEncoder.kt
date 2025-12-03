package com.panam.arm_hackthon.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * DA-CLIP Encoder Model
 * For image/text embeddings
 */
class DAClipEncoder(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    companion object {
        private const val MODEL_NAME = "daclip_encoder_full.onnx"
        private const val OLD_MODEL_NAME = "daclip_encoder.onnx"  // For cache cleanup
        private const val TAG = "DAClipEncoder"
    }

    /**
     * Initialize the model with Arm NNAPI optimization
     */
    suspend fun initialize(useNNAPI: Boolean = true): Boolean = withContext(Dispatchers.IO) {
        try {
            // Create ONNX Runtime environment
            ortEnv = OrtEnvironment.getEnvironment()

            // Clean up old cached model if it exists
            val oldModelFile = java.io.File(context.filesDir, OLD_MODEL_NAME)
            if (oldModelFile.exists()) {
                android.util.Log.i(TAG, "Deleting old cached model: $OLD_MODEL_NAME")
                oldModelFile.delete()
            }

            // Copy model from assets to internal storage (only once)
            val modelFile = java.io.File(context.filesDir, MODEL_NAME)
            if (!modelFile.exists()) {
                android.util.Log.i(TAG, "Copying model to internal storage...")
                context.assets.open("models/$MODEL_NAME").use { input ->
                    modelFile.outputStream().use { output ->
                        input.copyTo(output, bufferSize = 8192)
                    }
                }
                android.util.Log.i(TAG, "✓ Model copied: ${modelFile.length() / (1024*1024)} MB")
            } else {
                android.util.Log.i(TAG, "Model already exists: ${modelFile.length() / (1024*1024)} MB")
            }

            // Configure session options
            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                // Enable NNAPI for Arm acceleration
                if (useNNAPI) {
                    try {
                        addNnapi()
                        android.util.Log.i(TAG, "✓ NNAPI enabled (Arm acceleration)")
                    } catch (e: Exception) {
                        android.util.Log.w(TAG, "NNAPI not available, using CPU: ${e.message}")
                    }
                }

                setIntraOpNumThreads(4)
                setInterOpNumThreads(2)
            }

            // Create session from file path (ONNX loads efficiently from disk)
            ortSession = ortEnv?.createSession(modelFile.absolutePath, sessionOptions)

            // Log model info
            ortSession?.let { session ->
                android.util.Log.i(TAG, "✓ Model loaded successfully")
                android.util.Log.i(TAG, "Input names: ${session.inputNames}")
                android.util.Log.i(TAG, "Output names: ${session.outputNames}")

                // Log input/output shapes
                session.inputInfo.forEach { (name, info) ->
                    android.util.Log.i(TAG, "Input '$name': $info")
                }
                session.outputInfo.forEach { (name, info) ->
                    android.util.Log.i(TAG, "Output '$name': $info")
                }
            }

            true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to initialize", e)
            false
        }
    }

    /**
     * Encode an image to embeddings
     * Returns a map of output names to their corresponding embeddings
     */
    suspend fun encodeAll(
        bitmap: Bitmap,
        inputName: String = "image",
        targetSize: Int = 224
    ): Map<String, FloatArray>? = withContext(Dispatchers.IO) {
        val session = ortSession ?: run {
            android.util.Log.e(TAG, "Model not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Resize to target size (CLIP typically uses 224x224)
            val resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)

            android.util.Log.d(TAG, "Encoding ${targetSize}x${targetSize} image...")

            // Preprocess bitmap (CLIP typically uses ImageNet normalization)
            val inputData = preprocessBitmapForClip(resized)

            // Create ONNX tensor
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                inputData,
                longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())
            )

            // Run inference
            val inputs = mapOf(inputName to inputTensor)
            val outputs = session.run(inputs)

            // Get all output embeddings
            val result = mutableMapOf<String, FloatArray>()

            android.util.Log.d(TAG, "Number of outputs: ${outputs.size()}")

            outputs.forEach { (name, value) ->
                val outputTensor = value as OnnxTensor
                val embeddings = when (val tensorValue = outputTensor.value) {
                    is Array<*> -> (tensorValue[0] as FloatArray)
                    is FloatArray -> tensorValue
                    else -> {
                        val buffer = outputTensor.floatBuffer
                        FloatArray(buffer.remaining()).also { buffer.get(it) }
                    }
                }

                result[name] = embeddings
                android.util.Log.d(TAG, "Output '$name': ${embeddings.size} dims, sample: ${embeddings.take(3)}")
            }

            // Cleanup
            inputTensor.close()
            outputs.forEach { it.value.close() }

            val inferenceTime = System.currentTimeMillis() - startTime
            android.util.Log.i(TAG, "✓ Encoded in ${inferenceTime}ms")

            result

        } catch (e: Exception) {
            android.util.Log.e(TAG, "Encoding failed", e)
            null
        }
    }

    /**
     * Encode an image to embeddings (single output for backward compatibility)
     */
    suspend fun encode(
        bitmap: Bitmap,
        inputName: String = "image",
        outputName: String = "combined_features",
        targetSize: Int = 224
    ): FloatArray? = withContext(Dispatchers.IO) {
        val allOutputs = encodeAll(bitmap, inputName, targetSize)
        allOutputs?.get(outputName) ?: allOutputs?.values?.firstOrNull()
    }

    /**
     * Preprocess with ImageNet normalization (standard for CLIP)
     * Mean: [0.485, 0.456, 0.406]
     * Std: [0.229, 0.224, 0.225]
     */
    private fun preprocessBitmapForClip(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)

        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val buffer = FloatBuffer.allocate(3 * width * height)

        val meanR = 0.485f
        val meanG = 0.456f
        val meanB = 0.406f
        val stdR = 0.229f
        val stdG = 0.224f
        val stdB = 0.225f

        // R channel (normalized)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val r = ((pixel shr 16) and 0xFF) / 255f
                buffer.put((r - meanR) / stdR)
            }
        }

        // G channel (normalized)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val g = ((pixel shr 8) and 0xFF) / 255f
                buffer.put((g - meanG) / stdG)
            }
        }

        // B channel (normalized)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val b = (pixel and 0xFF) / 255f
                buffer.put((b - meanB) / stdB)
            }
        }

        buffer.rewind()
        return buffer
    }

    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnv = null
    }
}

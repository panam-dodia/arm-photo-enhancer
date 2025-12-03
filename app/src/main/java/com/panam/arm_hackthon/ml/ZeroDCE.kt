package com.panam.arm_hackthon.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * Zero-DCE (Zero-Reference Deep Curve Estimation)
 * Low-light image enhancement without reference images
 */
class ZeroDCE(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    companion object {
        private const val MODEL_NAME = "zerodce_iter8.onnx"
        private const val INPUT_NAME = "image"
        private const val OUTPUT_NAME = "output"
    }

    /**
     * Initialize the model with Arm NNAPI optimization
     */
    suspend fun initialize(useNNAPI: Boolean = true): Boolean = withContext(Dispatchers.IO) {
        try {
            // Create ONNX Runtime environment
            ortEnv = OrtEnvironment.getEnvironment()

            // Load model from assets
            val modelBytes = context.assets.open("models/$MODEL_NAME").use {
                it.readBytes()
            }

            // Configure session options
            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                // Enable NNAPI for Arm acceleration
                if (useNNAPI) {
                    try {
                        addNnapi()
                        android.util.Log.i("ZeroDCE", "✓ NNAPI enabled (Arm acceleration)")
                    } catch (e: Exception) {
                        android.util.Log.w("ZeroDCE", "NNAPI not available, using CPU: ${e.message}")
                    }
                }

                setIntraOpNumThreads(4)
                setInterOpNumThreads(2)
            }

            // Create session
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)

            // Log model info
            ortSession?.let { session ->
                val inputInfo = session.inputInfo.values.first()
                android.util.Log.i("ZeroDCE", "Model loaded: ${modelBytes.size / 1024} KB")
                android.util.Log.i("ZeroDCE", "Input: $inputInfo")
            }

            true
        } catch (e: Exception) {
            android.util.Log.e("ZeroDCE", "Failed to initialize", e)
            false
        }
    }

    /**
     * Enhance a low-light image
     */
    suspend fun enhance(bitmap: Bitmap, maxSize: Int = 512): Bitmap? = withContext(Dispatchers.IO) {
        val session = ortSession ?: run {
            android.util.Log.e("ZeroDCE", "Model not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Resize if needed
            val resized = resizeForInference(bitmap, maxSize)
            val width = resized.width
            val height = resized.height

            android.util.Log.d("ZeroDCE", "Processing ${width}x${height} image...")

            // Preprocess
            val inputData = preprocessBitmap(resized)

            // Create ONNX tensor
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                inputData,
                longArrayOf(1, 3, height.toLong(), width.toLong())
            )

            // Run inference
            val inputs = mapOf(INPUT_NAME to inputTensor)
            val outputs = session.run(inputs)

            // Get output
            val outputTensor = outputs[0] as OnnxTensor
            val outputData = outputTensor.floatBuffer

            // Postprocess
            val result = postprocessToBitmap(outputData, width, height)

            // Cleanup
            inputTensor.close()
            outputs.forEach { it.value.close() }

            val inferenceTime = System.currentTimeMillis() - startTime
            android.util.Log.i("ZeroDCE", "✓ Completed in ${inferenceTime}ms")

            result

        } catch (e: Exception) {
            android.util.Log.e("ZeroDCE", "Enhancement failed", e)
            null
        }
    }

    /**
     * Resize bitmap for inference with high-quality filtering
     */
    private fun resizeForInference(bitmap: Bitmap, maxSize: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        if (width <= maxSize && height <= maxSize) {
            return bitmap
        }

        val scale = minOf(
            maxSize.toFloat() / width,
            maxSize.toFloat() / height
        )

        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        // Use Canvas with high-quality Paint for smooth edges (no jagging)
        val result = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(result)

        val paint = android.graphics.Paint().apply {
            isAntiAlias = true       // Smooth edges
            isFilterBitmap = true    // Better interpolation
            isDither = true          // Reduce color banding
        }

        val srcRect = android.graphics.Rect(0, 0, width, height)
        val dstRect = android.graphics.Rect(0, 0, newWidth, newHeight)

        canvas.drawBitmap(bitmap, srcRect, dstRect, paint)

        return result
    }

    /**
     * Preprocess bitmap to model input format
     */
    private fun preprocessBitmap(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)

        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val buffer = FloatBuffer.allocate(3 * width * height)

        // R channel [0, 1]
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val r = ((pixel shr 16) and 0xFF) / 255f
                buffer.put(r)
            }
        }

        // G channel [0, 1]
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val g = ((pixel shr 8) and 0xFF) / 255f
                buffer.put(g)
            }
        }

        // B channel [0, 1]
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val b = (pixel and 0xFF) / 255f
                buffer.put(b)
            }
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Postprocess model output to bitmap
     */
    private fun postprocessToBitmap(buffer: FloatBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()

        val totalPixels = width * height

        val rChannel = FloatArray(totalPixels)
        val gChannel = FloatArray(totalPixels)
        val bChannel = FloatArray(totalPixels)

        buffer.get(rChannel)
        buffer.get(gChannel)
        buffer.get(bChannel)

        val pixels = IntArray(totalPixels)

        for (i in 0 until totalPixels) {
            val r = (rChannel[i].coerceIn(0f, 1f) * 255f).toInt()
            val g = (gChannel[i].coerceIn(0f, 1f) * 255f).toInt()
            val b = (bChannel[i].coerceIn(0f, 1f) * 255f).toInt()

            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * Release resources
     */
    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnv = null
    }
}
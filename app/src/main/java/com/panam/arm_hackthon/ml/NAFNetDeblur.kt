package com.panam.arm_hackthon.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * NAFNet Deblurring Model
 *
 * On-device image deblurring using NAFNet with Arm NNAPI optimization.
 * Model: NAFNet-GoPro-width32 (65.9 MB)
 */
class NAFNetDeblur(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    companion object {
        private const val MODEL_NAME = "nafnet_deblur.onnx"
        private const val INPUT_NAME = "input"
        private const val OUTPUT_NAME = "output"
    }

    /**
     * Initialize the model with Arm NNAPI optimization
     */
    suspend fun initialize(useNNAPI: Boolean = true) = withContext(Dispatchers.IO) {
        try {
            // Create ONNX Runtime environment
            ortEnv = OrtEnvironment.getEnvironment()

            // Load model from assets
            val modelBytes = context.assets.open("models/$MODEL_NAME").use {
                it.readBytes()
            }

            // Configure session options
            val sessionOptions = OrtSession.SessionOptions().apply {
                // Enable all optimizations
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                // Use NNAPI for Arm acceleration (GPU/NPU)
                if (useNNAPI) {
                    try {
                        addNnapi()
                        android.util.Log.i("NAFNetDeblur", "✓ NNAPI enabled (Arm GPU/NPU acceleration)")
                    } catch (e: Exception) {
                        android.util.Log.w("NAFNetDeblur", "NNAPI not available, using CPU: ${e.message}")
                    }
                }

                // Set thread count (optimize for mobile)
                setIntraOpNumThreads(4)
                setInterOpNumThreads(2)
            }

            // Create session
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)

            // Log model info
            ortSession?.let { session ->
                val inputInfo = session.inputInfo.values.first()
                android.util.Log.i("NAFNetDeblur", "Model loaded: ${modelBytes.size / (1024*1024)} MB")
                android.util.Log.i("NAFNetDeblur", "Input: $inputInfo")
            }

            true
        } catch (e: Exception) {
            android.util.Log.e("NAFNetDeblur", "Failed to initialize model", e)
            false
        }
    }

    /**
     * Deblur an image using NAFNet
     */
    suspend fun deblur(bitmap: Bitmap, maxSize: Int = 512): Bitmap? = withContext(Dispatchers.IO) {
        val session = ortSession ?: run {
            android.util.Log.e("NAFNetDeblur", "Model not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Resize if needed
            val resized = resizeForInference(bitmap, maxSize)
            val width = resized.width
            val height = resized.height

            android.util.Log.d("NAFNetDeblur", "Processing ${width}x${height} image...")

            // Preprocess
            val inputData = preprocessBitmap(resized)

            // DEBUG: Check input values
            inputData.rewind()
            val inputSample = FloatArray(30)
            inputData.get(inputSample)
            inputData.rewind()
            android.util.Log.d("NAFNetDeblur", "INPUT First 10 R values: ${inputSample.take(10)}")
            android.util.Log.d("NAFNetDeblur", "INPUT Min/Max: ${inputSample.minOrNull()} / ${inputSample.maxOrNull()}")

            // Create ONNX tensor
            val inputTensor = OnnxTensor.createTensor(
                ortEnv!!,
                inputData,
                longArrayOf(1, 3, height.toLong(), width.toLong())
            )

            // Run inference
            val inferenceStart = System.currentTimeMillis()
            val inputs = mapOf(INPUT_NAME to inputTensor)
            val outputs = session.run(inputs)
            android.util.Log.d("NAFNetDeblur", "Inference: ${System.currentTimeMillis() - inferenceStart}ms")

            // Get output
            val outputTensor = outputs[0] as OnnxTensor
            val outputData = outputTensor.floatBuffer

            // DEBUG: Check output values
            outputData.rewind()
            val outputSample = FloatArray(30)
            outputData.get(outputSample)
            outputData.rewind()
            android.util.Log.d("NAFNetDeblur", "OUTPUT First 10 R values: ${outputSample.take(10)}")
            android.util.Log.d("NAFNetDeblur", "OUTPUT Min/Max: ${outputSample.minOrNull()} / ${outputSample.maxOrNull()}")

            // Check if output is same as input (model not working)
            val inputCheck = inputSample.take(10)
            val outputCheck = outputSample.take(10)
            val isSame = inputCheck.zip(outputCheck).all { (a, b) -> Math.abs(a - b) < 0.01f }
            if (isSame) {
                android.util.Log.e("NAFNetDeblur", "⚠️ OUTPUT = INPUT! Model not processing correctly!")
            }

            // Postprocess
            val result = postprocessToBitmap(outputData, width, height)

            // Cleanup
            inputTensor.close()
            outputs.forEach { it.value.close() }

            val inferenceTime = System.currentTimeMillis() - startTime
            android.util.Log.i("NAFNetDeblur", "✓ Completed in ${inferenceTime}ms")

            result

        } catch (e: Exception) {
            android.util.Log.e("NAFNetDeblur", "Deblurring failed", e)
            null
        }
    }

    /**
     * Resize bitmap for inference (maintain aspect ratio)
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

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }

    /**
     * Preprocess bitmap to model input format
     * NAFNet might expect [0, 255] range instead of [0, 1]
     */
    private fun preprocessBitmap(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)

        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val buffer = FloatBuffer.allocate(3 * width * height)

        // Try [0, 255] range instead of [0, 1]
        // R channel
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val r = ((pixel shr 16) and 0xFF).toFloat()  // Keep as 0-255
                buffer.put(r)
            }
        }

        // G channel
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val g = ((pixel shr 8) and 0xFF).toFloat()  // Keep as 0-255
                buffer.put(g)
            }
        }

        // B channel
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val b = (pixel and 0xFF).toFloat()  // Keep as 0-255
                buffer.put(b)
            }
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Postprocess model output to bitmap
     * Handle output that might be in [0, 255] range
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
            // Clamp to [0, 255] range (model outputs in this range)
            val r = rChannel[i].coerceIn(0f, 255f).toInt()
            val g = gChannel[i].coerceIn(0f, 255f).toInt()
            val b = bChannel[i].coerceIn(0f, 255f).toInt()

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
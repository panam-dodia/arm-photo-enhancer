package com.panam.arm_hackthon.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * U-Net Model for image-to-image tasks
 */
class UNet(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // SDE schedule (initialized once)
    private var thetas: FloatArray? = null
    private var thetasCumsum: FloatArray? = null
    private var dt: Float = 0f
    private var sigmas: FloatArray? = null
    private var sigmaBars: FloatArray? = null

    companion object {
        private const val MODEL_NAME = "unet.onnx"
        private const val TAG = "UNet"

        // SDE parameters (matching Python script)
        private const val DEFAULT_T = 100
        private const val DEFAULT_MAX_SIGMA = 50f / 255f  // Normalize to [0,1] range
        private const val DEFAULT_EPS = 0.005f
    }

    /**
     * Initialize the model with Arm NNAPI optimization
     */
    suspend fun initialize(useNNAPI: Boolean = true): Boolean = withContext(Dispatchers.IO) {
        try {
            // Create ONNX Runtime environment
            ortEnv = OrtEnvironment.getEnvironment()

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

                // Optimal thread count for 384px images
                setIntraOpNumThreads(4)  // Restored for better performance
                setInterOpNumThreads(2)  // Restored for better performance
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

            // Initialize SDE schedule
            initSchedule(DEFAULT_T, DEFAULT_MAX_SIGMA, DEFAULT_EPS)

            true
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Failed to initialize", e)
            false
        }
    }

    /**
     * Initialize SDE noise schedule (cosine schedule)
     * Matches the Python implementation exactly
     */
    private fun initSchedule(T: Int, maxSigma: Float, eps: Float) {
        val timesteps = T + 2
        val steps = timesteps + 1
        val s = 0.008f

        // Cosine schedule
        val x = FloatArray(steps) { it.toFloat() }
        val alphasCumprod = FloatArray(steps) { i ->
            val value = ((x[i] / timesteps) + s) / (1 + s) * Math.PI.toFloat() * 0.5f
            (kotlin.math.cos(value) * kotlin.math.cos(value))
        }
        // Normalize
        val alpha0 = alphasCumprod[0]
        for (i in alphasCumprod.indices) {
            alphasCumprod[i] /= alpha0
        }

        // Thetas = 1 - alphas_cumprod[1:-1]
        thetas = FloatArray(T) { i ->
            1 - alphasCumprod[i + 1]
        }

        // Cumulative sum
        thetasCumsum = FloatArray(T) { i ->
            thetas!!.take(i + 1).sum() - thetas!![0]
        }

        // dt = -1 / thetas_cumsum[-1] * log(eps)
        dt = -1.0f / thetasCumsum!!.last() * kotlin.math.ln(eps)

        // sigmas = sqrt(max_sigma^2 * 2 * thetas)
        sigmas = FloatArray(T) { i ->
            kotlin.math.sqrt(maxSigma * maxSigma * 2 * thetas!![i])
        }

        // sigma_bars = sqrt(max_sigma^2 * (1 - exp(-2 * thetas_cumsum * dt)))
        sigmaBars = FloatArray(T) { i ->
            val expTerm = kotlin.math.exp(-2 * thetasCumsum!![i] * dt)
            kotlin.math.sqrt(maxSigma * maxSigma * (1 - expTerm))
        }

        android.util.Log.i(TAG, "SDE schedule initialized: T=$T, max_sigma=$maxSigma, dt=$dt")
    }

    /**
     * Restore image using iterative SDE diffusion with CLIP guidance
     * Uses proper reverse SDE with multiple denoising steps
     *
     * @param bitmap Input degraded image
     * @param imageContext Image features from CLIP (512 dims)
     * @param degraContext Degradation features from CLIP (512 dims)
     * @param maxSize Maximum image dimension
     * @param numSteps Number of diffusion steps (default 5)
     * @param onProgress Callback for progress updates (step, totalSteps)
     */
    suspend fun restoreWithCLIP(
        bitmap: Bitmap,
        imageContext: FloatArray,
        degraContext: FloatArray,
        maxSize: Int = 512,
        numSteps: Int = 5,
        onProgress: ((Int, Int) -> Unit)? = null
    ): Bitmap? = withContext(Dispatchers.IO) {
        val session = ortSession ?: run {
            android.util.Log.e(TAG, "Model not initialized")
            return@withContext null
        }

        if (sigmaBars == null) {
            android.util.Log.e(TAG, "SDE schedule not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Resize if needed
            val resized = resizeForInference(bitmap, maxSize)
            val width = resized.width
            val height = resized.height
            val pixelsPerChannel = width * height

            android.util.Log.i(TAG, "Starting SDE restoration: ${width}x${height}, steps=$numSteps")

            // Preprocess bitmap to FloatBuffer (CHW format, 0-1 range)
            val lqImage = preprocessBitmap(resized)

            // Step 1: Add initial noise (matches Python: x = lq + randn * max_sigma)
            val x = addInitialNoise(lqImage, pixelsPerChannel)
            android.util.Log.d(TAG, "Initial noisy state created")

            // Prepare context tensors (reused across iterations)
            val imageContextTensor = OnnxTensor.createTensor(
                ortEnv!!,
                FloatBuffer.wrap(imageContext),
                longArrayOf(1, imageContext.size.toLong())
            )
            val degraContextTensor = OnnxTensor.createTensor(
                ortEnv!!,
                FloatBuffer.wrap(degraContext),
                longArrayOf(1, degraContext.size.toLong())
            )

            // Step 2: Iterative SDE reverse diffusion
            val sampleScale = DEFAULT_T.toFloat() / numSteps
            val random = java.util.Random()

            for (step in numSteps downTo 1) {
                // Map step to actual timestep index
                val t = (step * sampleScale).toInt().coerceIn(0, DEFAULT_T - 1)

                // Report progress (convert to forward counting: 1, 2, 3...)
                val currentStep = numSteps - step + 1
                onProgress?.invoke(currentStep, numSteps)

                // Log detailed timing for first step (usually the slowest)
                val stepStartTime = if (currentStep == 1) System.currentTimeMillis() else 0L
                if (currentStep == 1) {
                    android.util.Log.i(TAG, "Starting first UNet inference (384px with NNAPI)...")
                }

                // Run UNet to get noise prediction
                val noisePrediction = runUNetStep(
                    session, x, lqImage, t.toLong(),
                    imageContextTensor, degraContextTensor,
                    width, height
                )

                if (currentStep == 1) {
                    val firstStepTime = System.currentTimeMillis() - stepStartTime
                    android.util.Log.i(TAG, "✓ First step completed in ${firstStepTime}ms")
                }

                // Convert noise to score: score = -noise / sigma_bars[t]
                val score = FloatBuffer.allocate(pixelsPerChannel * 3)
                noisePrediction.rewind()
                val sigmaBar = sigmaBars!![t]
                for (i in 0 until pixelsPerChannel * 3) {
                    score.put(-noisePrediction.get() / sigmaBar)
                }
                score.rewind()
                noisePrediction.rewind()

                // Perform reverse SDE step
                sdeReverseStep(x, score, lqImage, t, pixelsPerChannel, random)

                android.util.Log.d(TAG, "Completed step $currentStep/$numSteps (t=$t)")
            }

            // Cleanup context tensors
            imageContextTensor.close()
            degraContextTensor.close()

            // Step 3: Convert final result to bitmap
            android.util.Log.i(TAG, "Converting result to bitmap...")

            val result = try {
                postprocessToBitmap(x, width, height)
            } catch (e: OutOfMemoryError) {
                android.util.Log.e(TAG, "Out of memory during bitmap conversion", e)
                // Try to free memory and retry once
                System.gc()
                Thread.sleep(100)
                try {
                    postprocessToBitmap(x, width, height)
                } catch (e2: OutOfMemoryError) {
                    android.util.Log.e(TAG, "Out of memory on retry", e2)
                    return@withContext null
                }
            }

            val totalTime = System.currentTimeMillis() - startTime
            android.util.Log.i(TAG, "✓ SDE restoration complete in ${totalTime}ms (${totalTime/numSteps}ms/step)")

            result

        } catch (e: Exception) {
            android.util.Log.e(TAG, "SDE restoration failed", e)
            e.printStackTrace()
            null
        }
    }

    /**
     * Add initial noise to create starting state for diffusion
     * Formula: x = lq + randn * max_sigma
     */
    private fun addInitialNoise(lqImage: FloatBuffer, pixelsPerChannel: Int): FloatBuffer {
        lqImage.rewind()
        val x = FloatBuffer.allocate(pixelsPerChannel * 3)
        val random = java.util.Random()

        for (i in 0 until pixelsPerChannel * 3) {
            val lq = lqImage.get()
            val noise = random.nextGaussian().toFloat() * DEFAULT_MAX_SIGMA
            x.put(lq + noise)
        }

        x.rewind()
        lqImage.rewind()
        return x
    }

    /**
     * Run a single UNet denoising step
     */
    private fun runUNetStep(
        session: OrtSession,
        noisyImage: FloatBuffer,
        lqImage: FloatBuffer,
        timestep: Long,
        imageContextTensor: OnnxTensor,
        degraContextTensor: OnnxTensor,
        width: Int,
        height: Int
    ): FloatBuffer {
        noisyImage.rewind()
        lqImage.rewind()

        // Create tensors for this step
        val noisyTensor = OnnxTensor.createTensor(
            ortEnv!!,
            noisyImage,
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )
        val lqTensor = OnnxTensor.createTensor(
            ortEnv!!,
            lqImage,
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )
        val timestepTensor = OnnxTensor.createTensor(
            ortEnv!!,
            LongBuffer.wrap(longArrayOf(timestep)),
            longArrayOf(1)
        )

        // Run inference
        val inputs = mapOf(
            "noisy_image" to noisyTensor,
            "lq_image" to lqTensor,
            "timestep" to timestepTensor,
            "image_context" to imageContextTensor,
            "degra_context" to degraContextTensor
        )

        val outputs = session.run(inputs)
        val noiseTensor = outputs[0] as OnnxTensor

        // Copy the noise prediction to avoid memory issues after closing
        val noisePrediction = FloatBuffer.allocate(width * height * 3)
        val tempBuffer = noiseTensor.floatBuffer
        tempBuffer.rewind()
        noisePrediction.put(tempBuffer)
        noisePrediction.rewind()

        // Cleanup step tensors immediately
        noisyTensor.close()
        lqTensor.close()
        timestepTensor.close()
        outputs.close()

        noisyImage.rewind()
        lqImage.rewind()

        return noisePrediction
    }

    /**
     * Perform reverse SDE step
     * Formula: x = x - reverse_drift - dispersion
     * where:
     *   reverse_drift = (theta * (mu - x) - sigma^2 * score) * dt
     *   dispersion = sigma * randn * sqrt(dt)
     */
    private fun sdeReverseStep(
        x: FloatBuffer,
        score: FloatBuffer,
        mu: FloatBuffer,
        t: Int,
        pixelsPerChannel: Int,
        random: java.util.Random
    ) {
        val theta = thetas!![t]
        val sigma = sigmas!![t]
        val sqrtDt = kotlin.math.sqrt(dt)

        x.rewind()
        score.rewind()
        mu.rewind()

        for (i in 0 until pixelsPerChannel * 3) {
            val xVal = x.get()
            val scoreVal = score.get()
            val muVal = mu.get()

            // Reverse drift
            val reverseDrift = (theta * (muVal - xVal) - sigma * sigma * scoreVal) * dt

            // Dispersion (stochastic term)
            val dispersion = sigma * random.nextGaussian().toFloat() * sqrtDt

            // Update x
            val newX = xVal - reverseDrift - dispersion
            x.put(i, newX)
        }

        x.rewind()
        score.rewind()
        mu.rewind()
    }

    /**
     * Resize image if larger than maxSize, preserving aspect ratio
     * No padding - just resize down if needed
     * Uses high-quality Canvas-based scaling for smooth edges
     */
    private fun resizeForInference(bitmap: Bitmap, maxSize: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // If image is larger than maxSize, resize it down
        return if (width > maxSize || height > maxSize) {
            val scale = minOf(
                maxSize.toFloat() / width,
                maxSize.toFloat() / height
            )
            val newWidth = (width * scale).toInt()
            val newHeight = (height * scale).toInt()

            android.util.Log.d(TAG, "Resizing ${width}x${height} → ${newWidth}x${newHeight} (no padding)")

            // Use Canvas with high-quality Paint for smooth edges (no jagging)
            val result = android.graphics.Bitmap.createBitmap(newWidth, newHeight, android.graphics.Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(result)

            val paint = android.graphics.Paint().apply {
                isAntiAlias = true       // Smooth edges
                isFilterBitmap = true    // Better interpolation
                isDither = true          // Reduce color banding
            }

            val srcRect = android.graphics.Rect(0, 0, width, height)
            val dstRect = android.graphics.Rect(0, 0, newWidth, newHeight)

            canvas.drawBitmap(bitmap, srcRect, dstRect, paint)

            result
        } else {
            android.util.Log.d(TAG, "Image ${width}x${height} fits within ${maxSize}px, no resize needed")
            bitmap
        }
    }

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

    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnv = null
    }
}

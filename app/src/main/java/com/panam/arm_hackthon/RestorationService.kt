package com.panam.arm_hackthon

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import com.panam.arm_hackthon.ml.UNet
import com.panam.arm_hackthon.ml.DAClipEncoder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class RestorationService : LifecycleService() {

    private var unet: UNet? = null
    private var daclipEncoder: DAClipEncoder? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var modelsReady = false

    // Pending restoration request (set by onStartCommand, processed when models ready)
    private var pendingImagePath: String? = null
    private var pendingNumSteps: Int = 100

    private val notificationChannelId = "restoration_service_channel"
    private val notificationId = 2

    companion object {
        const val ACTION_START_RESTORATION = "com.panam.arm_hackthon.START_RESTORATION"
        const val ACTION_STOP_RESTORATION = "com.panam.arm_hackthon.STOP_RESTORATION"

        const val EXTRA_IMAGE_PATH = "image_path"
        const val EXTRA_NUM_STEPS = "num_steps"

        const val BROADCAST_PROGRESS = "com.panam.arm_hackthon.RESTORATION_PROGRESS"
        const val BROADCAST_COMPLETE = "com.panam.arm_hackthon.RESTORATION_COMPLETE"
        const val BROADCAST_ERROR = "com.panam.arm_hackthon.RESTORATION_ERROR"

        const val EXTRA_CURRENT_STEP = "current_step"
        const val EXTRA_TOTAL_STEPS = "total_steps"
        const val EXTRA_OUTPUT_PATH = "output_path"
        const val EXTRA_ERROR_MESSAGE = "error_message"
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()

        // Start foreground IMMEDIATELY to prevent Android from killing the service
        startForeground(notificationId, createNotification("Initializing models...", 0, 100))

        // Initialize models
        unet = UNet(this)
        daclipEncoder = DAClipEncoder(this)

        lifecycleScope.launch {
            // Initialize models with NNAPI for Arm acceleration
            val unetSuccess = unet?.initialize(useNNAPI = true) ?: false
            val clipSuccess = daclipEncoder?.initialize(useNNAPI = true) ?: false
            modelsReady = unetSuccess && clipSuccess
            android.util.Log.i("RestorationService", "Models ready: $modelsReady (UNet: NNAPI, CLIP: NNAPI)")

            // If models loaded successfully and there's a pending restoration, start it
            if (modelsReady) {
                startPendingRestoration()
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)

        when (intent?.action) {
            ACTION_START_RESTORATION -> {
                val imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
                val numSteps = intent.getIntExtra(EXTRA_NUM_STEPS, 100)

                if (imagePath != null) {
                    // Save the restoration request
                    pendingImagePath = imagePath
                    pendingNumSteps = numSteps

                    android.util.Log.i("RestorationService", "Restoration request received, waiting for models...")

                    // If models are already ready, start immediately
                    // Otherwise, it will start automatically when models finish loading
                    if (modelsReady) {
                        startPendingRestoration()
                    }
                }
            }
            ACTION_STOP_RESTORATION -> {
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = "Image Restoration Service"
            val descriptionText = "Shows progress of image restoration"
            val importance = NotificationManager.IMPORTANCE_LOW
            val channel = NotificationChannel(notificationChannelId, name, importance).apply {
                description = descriptionText
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(text: String, current: Int, total: Int): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = NotificationCompat.Builder(this, notificationChannelId)
            .setContentTitle("Image Restoration")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)

        if (total > 0) {
            builder.setProgress(total, current, false)
        }

        return builder.build()
    }

    /**
     * Check if there's a pending restoration request and start it
     * Called when models finish loading or when request arrives after models are ready
     */
    private fun startPendingRestoration() {
        val imagePath = pendingImagePath
        val numSteps = pendingNumSteps

        if (imagePath != null) {
            // Clear pending request
            pendingImagePath = null

            // Update notification
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(notificationId, createNotification("Starting restoration...", 0, numSteps))

            android.util.Log.i("RestorationService", "Starting pending restoration...")
            startRestoration(imagePath, numSteps)
        }
    }

    private fun startRestoration(imagePath: String, numSteps: Int) {
        lifecycleScope.launch {
            // Acquire WakeLock (384px @ 100 steps can take 30-45 minutes)
            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "ArmHackthon::RestorationServiceWakeLock"
            )
            wakeLock?.acquire(60 * 60 * 1000L) // 60 minutes max for 384px restoration

            try {
                // Load image
                val bitmap = android.graphics.BitmapFactory.decodeFile(imagePath)
                if (bitmap == null) {
                    broadcastError("Failed to load image")
                    stopSelf()
                    return@launch
                }

                // Get CLIP embeddings
                val clipOutputs = withContext(Dispatchers.IO) {
                    daclipEncoder?.encodeAll(bitmap)
                }

                if (clipOutputs == null || clipOutputs.isEmpty()) {
                    broadcastError("CLIP encoding failed")
                    stopSelf()
                    return@launch
                }

                // Extract contexts
                val combined = clipOutputs["combined_features"] ?: clipOutputs.values.first()
                val imageContext: FloatArray
                val degraContext: FloatArray

                when (combined.size) {
                    1024 -> {
                        imageContext = combined.copyOfRange(0, 512)
                        degraContext = combined.copyOfRange(512, 1024)
                    }
                    512 -> {
                        imageContext = combined
                        degraContext = combined.copyOf()
                    }
                    else -> {
                        broadcastError("Unexpected CLIP dimensions: ${combined.size}")
                        stopSelf()
                        return@launch
                    }
                }

                // CRITICAL: Close DAClipEncoder to free memory (700MB)
                // We don't need it anymore after getting embeddings
                daclipEncoder?.close()
                daclipEncoder = null
                android.util.Log.i("RestorationService", "✓ DAClipEncoder closed to free memory")

                // Force garbage collection to reclaim memory before heavy UNet processing
                System.gc()
                android.util.Log.i("RestorationService", "Memory freed before UNet inference")

                // Log memory before starting
                val runtime = Runtime.getRuntime()
                val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
                val maxMemory = runtime.maxMemory() / (1024 * 1024)
                android.util.Log.i("RestorationService", "Memory before UNet: ${usedMemory}MB / ${maxMemory}MB")

                // Update notification before starting to show activity
                val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(
                    notificationId,
                    createNotification("Preparing first step (15-30s)...", 0, numSteps)
                )

                // Restore image with error handling
                val restored = try {
                    withContext(Dispatchers.IO) {
                        unet?.restoreWithCLIP(
                            bitmap, imageContext, degraContext,
                            maxSize = 384,  // Working size - DO NOT increase or Android kills the process
                            numSteps = numSteps
                        ) { currentStep, totalSteps ->
                            // Update notification and broadcast progress
                            val notification = createNotification(
                                "Step $currentStep/$totalSteps",
                                currentStep,
                                totalSteps
                            )
                            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                            notificationManager.notify(notificationId, notification)

                            broadcastProgress(currentStep, totalSteps)
                        }
                    }
                } catch (e: OutOfMemoryError) {
                    android.util.Log.e("RestorationService", "Out of memory during restoration", e)
                    broadcastError("Out of memory - try closing other apps")
                    null
                } catch (e: Exception) {
                    android.util.Log.e("RestorationService", "UNet restoration crashed", e)
                    broadcastError("Restoration crashed: ${e.javaClass.simpleName}")
                    null
                }

                if (restored != null) {
                    try {
                        android.util.Log.i("RestorationService", "Saving restored image...")

                        // Save output
                        val outputFile = File(cacheDir, "restored_${System.currentTimeMillis()}.png")
                        FileOutputStream(outputFile).use { out ->
                            val compressed = restored.compress(Bitmap.CompressFormat.PNG, 100, out)
                            if (!compressed) {
                                android.util.Log.e("RestorationService", "Failed to compress bitmap")
                                broadcastError("Failed to save result")
                                return@launch
                            }
                        }

                        android.util.Log.i("RestorationService", "✓ Result saved: ${outputFile.length() / 1024} KB")
                        broadcastComplete(outputFile.absolutePath)
                    } catch (e: OutOfMemoryError) {
                        android.util.Log.e("RestorationService", "Out of memory while saving result", e)
                        broadcastError("Out of memory while saving - result too large")
                    } catch (e: Exception) {
                        android.util.Log.e("RestorationService", "Failed to save result", e)
                        broadcastError("Failed to save result: ${e.message}")
                    }
                } else {
                    android.util.Log.e("RestorationService", "Restored bitmap is null")
                    broadcastError("Restoration failed - result is empty")
                }

            } catch (e: Exception) {
                android.util.Log.e("RestorationService", "Restoration error", e)
                broadcastError(e.message ?: "Unknown error")
            } finally {
                wakeLock?.release()
                stopSelf()
            }
        }
    }

    private fun broadcastProgress(current: Int, total: Int) {
        val intent = Intent(BROADCAST_PROGRESS).apply {
            setPackage(packageName)  // Make broadcast explicit for Android 8.0+
            putExtra(EXTRA_CURRENT_STEP, current)
            putExtra(EXTRA_TOTAL_STEPS, total)
        }
        sendBroadcast(intent)
    }

    private fun broadcastComplete(outputPath: String) {
        android.util.Log.i("RestorationService", "Broadcasting completion: $outputPath")
        val intent = Intent(BROADCAST_COMPLETE).apply {
            setPackage(packageName)  // Make broadcast explicit for Android 8.0+
            putExtra(EXTRA_OUTPUT_PATH, outputPath)
        }
        sendBroadcast(intent)
        android.util.Log.i("RestorationService", "Broadcast sent, stopping service")
    }

    private fun broadcastError(message: String) {
        android.util.Log.e("RestorationService", "Broadcasting error: $message")
        val intent = Intent(BROADCAST_ERROR).apply {
            setPackage(packageName)  // Make broadcast explicit for Android 8.0+
            putExtra(EXTRA_ERROR_MESSAGE, message)
        }
        sendBroadcast(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
            }
        }
        unet?.close()
        daclipEncoder?.close() // Safe to call even if already closed/null
    }

    override fun onBind(intent: Intent): IBinder? {
        super.onBind(intent)
        return null
    }
}

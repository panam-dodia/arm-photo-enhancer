package com.panam.arm_hackthon

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Build
import android.provider.MediaStore
import android.widget.*
import androidx.exifinterface.media.ExifInterface
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.panam.arm_hackthon.ml.ZeroDCE
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException
import android.content.BroadcastReceiver
import android.content.IntentFilter
import android.content.Intent
import java.io.File
import java.io.FileOutputStream
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import androidx.core.app.NotificationCompat

class MainActivity : AppCompatActivity() {

    private lateinit var comparisonSlider: com.panam.arm_hackthon.views.ImageComparisonSlider
    private lateinit var selectImageButton: Button
    private lateinit var enhanceButton: Button
    private lateinit var restoreButton: Button
    private lateinit var saveButton: Button
    private lateinit var statusText: TextView
    private lateinit var performanceText: TextView
    private lateinit var sliderInstructions: TextView
    private lateinit var progressBar: ProgressBar

    private var zeroDCE: ZeroDCE? = null
    private var currentInputBitmap: Bitmap? = null
    private var currentOutputBitmap: Bitmap? = null

    private var restorationReceiver: BroadcastReceiver? = null

    companion object {
        private const val SAVE_NOTIFICATION_CHANNEL_ID = "image_save_channel"
        private const val SAVE_NOTIFICATION_ID = 1
    }

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            loadImage(it)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        comparisonSlider = findViewById(R.id.comparisonSlider)
        selectImageButton = findViewById(R.id.selectImageButton)
        enhanceButton = findViewById(R.id.enhanceButton)
        restoreButton = findViewById(R.id.restoreButton)
        saveButton = findViewById(R.id.saveButton)
        statusText = findViewById(R.id.statusText)
        performanceText = findViewById(R.id.performanceText)
        sliderInstructions = findViewById(R.id.sliderInstructions)
        progressBar = findViewById(R.id.progressBar)

        // Create notification channel for save notifications
        createSaveNotificationChannel()

        // Initialize Zero-DCE model for enhancement (lazy loading for restoration models)
        zeroDCE = ZeroDCE(this)

        lifecycleScope.launch {
            statusText.text = "Initializing enhancement model..."

            // Initialize only Zero-DCE (UNet and DAClipEncoder load in service when needed)
            val zeroDCESuccess = zeroDCE?.initialize(useNNAPI = true) ?: false

            if (zeroDCESuccess) {
                statusText.text = "Ready - Select an image"
                enhanceButton.isEnabled = false // Enable after image selection
                restoreButton.isEnabled = false // Enable after image selection
            } else {
                statusText.text = "Model initialization failed"
                enhanceButton.isEnabled = false
                restoreButton.isEnabled = false
            }
        }

        // Set up button listeners
        selectImageButton.setOnClickListener {
            imagePickerLauncher.launch("image/*")
        }

        enhanceButton.setOnClickListener {
            currentInputBitmap?.let { bitmap ->
                enhanceImage(bitmap)
            }
        }

        restoreButton.setOnClickListener {
            currentInputBitmap?.let { bitmap ->
                restoreImage(bitmap)
            }
        }

        saveButton.setOnClickListener {
            currentOutputBitmap?.let { bitmap ->
                saveImageToGallery(bitmap)
            }
        }

        // Initially disable buttons
        enhanceButton.isEnabled = false
        restoreButton.isEnabled = false
        saveButton.isEnabled = false
    }

    private fun loadImage(uri: Uri) {
        try {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)

            // Fix EXIF orientation (rotates images that are incorrectly oriented)
            val rotatedBitmap = fixImageOrientation(uri, bitmap)
            currentInputBitmap = rotatedBitmap

            // Set before image in slider
            comparisonSlider.setBeforeImage(rotatedBitmap)
            comparisonSlider.setAfterImage(rotatedBitmap) // Show same image initially
            comparisonSlider.reset()

            enhanceButton.isEnabled = true
            restoreButton.isEnabled = true
            saveButton.isEnabled = false
            sliderInstructions.visibility = TextView.GONE
            statusText.text = "Ready - Select an operation"
            performanceText.text = ""
            performanceText.visibility = TextView.GONE
        } catch (e: IOException) {
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
        }
    }

    private fun fixImageOrientation(uri: Uri, bitmap: Bitmap): Bitmap {
        val inputStream = contentResolver.openInputStream(uri) ?: return bitmap

        return try {
            val exif = ExifInterface(inputStream)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )

            android.util.Log.i("MainActivity", "EXIF orientation: $orientation")

            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> {
                    android.util.Log.i("MainActivity", "Rotating image 90°")
                    rotateBitmap(bitmap, 90f)
                }
                ExifInterface.ORIENTATION_ROTATE_180 -> {
                    android.util.Log.i("MainActivity", "Rotating image 180°")
                    rotateBitmap(bitmap, 180f)
                }
                ExifInterface.ORIENTATION_ROTATE_270 -> {
                    android.util.Log.i("MainActivity", "Rotating image 270°")
                    rotateBitmap(bitmap, 270f)
                }
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
                    android.util.Log.i("MainActivity", "Flipping image horizontally")
                    flipBitmap(bitmap, horizontal = true)
                }
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                    android.util.Log.i("MainActivity", "Flipping image vertically")
                    flipBitmap(bitmap, horizontal = false)
                }
                else -> {
                    android.util.Log.i("MainActivity", "No rotation needed")
                    bitmap
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "Failed to read EXIF", e)
            bitmap
        } finally {
            inputStream.close()
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun flipBitmap(bitmap: Bitmap, horizontal: Boolean): Bitmap {
        val matrix = Matrix().apply {
            if (horizontal) postScale(-1f, 1f)
            else postScale(1f, -1f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun enhanceImage(bitmap: Bitmap) {
        lifecycleScope.launch {
            try {
                // Show progress
                progressBar.visibility = ProgressBar.VISIBLE
                enhanceButton.isEnabled = false
                statusText.text = "Processing..."
                performanceText.text = ""

                // Measure time
                val startTime = System.currentTimeMillis()

                // Enhance image on IO thread
                val enhanced = withContext(Dispatchers.IO) {
                    zeroDCE?.enhance(bitmap)
                }

                val inferenceTime = System.currentTimeMillis() - startTime

                // Update UI on main thread
                if (enhanced != null) {
                    currentOutputBitmap = enhanced

                    // Set after image in slider
                    comparisonSlider.setAfterImage(enhanced)
                    comparisonSlider.reset() // Reset to middle position

                    statusText.text = "Enhancement complete"
                    sliderInstructions.visibility = TextView.VISIBLE

                    saveButton.isEnabled = true
                } else {
                    statusText.text = "Failed"
                    Toast.makeText(
                        this@MainActivity,
                        "Enhancement failed",
                        Toast.LENGTH_SHORT
                    ).show()
                }

            } catch (e: Exception) {
                statusText.text = "Error"
                Toast.makeText(
                    this@MainActivity,
                    "Error: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
            } finally {
                progressBar.visibility = ProgressBar.GONE
                enhanceButton.isEnabled = true
            }
        }
    }

    private fun createSaveNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = "Image Saved"
            val descriptionText = "Notifications when images are saved"
            val importance = NotificationManager.IMPORTANCE_DEFAULT
            val channel = NotificationChannel(SAVE_NOTIFICATION_CHANNEL_ID, name, importance).apply {
                description = descriptionText
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun showSaveNotification(imageUri: Uri, filename: String) {
        // Intent to open the saved image
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(imageUri, "image/jpeg")
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION
        }

        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, SAVE_NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Image Saved")
            .setContentText("Tap to view $filename")
            .setSmallIcon(android.R.drawable.ic_menu_gallery)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(SAVE_NOTIFICATION_ID, notification)
    }

    private fun saveImageToGallery(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val filename = "enhanced_${System.currentTimeMillis()}.jpg"
                val contentValues = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ArmEnhancer")
                }

                val uri = contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    contentValues
                )

                uri?.let {
                    contentResolver.openOutputStream(it)?.use { stream ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
                    }

                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@MainActivity,
                            "Saved to Gallery",
                            Toast.LENGTH_SHORT
                        ).show()

                        // Show notification with option to view image
                        showSaveNotification(it, filename)
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to save: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    private fun restoreImage(bitmap: Bitmap) {
        try {
            // Save bitmap to temporary file
            val tempFile = File(cacheDir, "input_${System.currentTimeMillis()}.png")
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            // Show progress in main UI (user can minimize app)
            restoreButton.isEnabled = false
            enhanceButton.isEnabled = false
            selectImageButton.isEnabled = false
            progressBar.visibility = ProgressBar.VISIBLE
            statusText.text = "Starting restoration..."
            performanceText.visibility = TextView.VISIBLE
            performanceText.text = "You can minimize the app during restoration"

            // Start foreground service
            val serviceIntent = Intent(this, RestorationService::class.java).apply {
                action = RestorationService.ACTION_START_RESTORATION
                putExtra(RestorationService.EXTRA_IMAGE_PATH, tempFile.absolutePath)
                putExtra(RestorationService.EXTRA_NUM_STEPS, 100)
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(serviceIntent)
            } else {
                startService(serviceIntent)
            }

        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "Failed to start restoration", e)
            Toast.makeText(this, "Failed to start restoration: ${e.message}", Toast.LENGTH_SHORT).show()

            restoreButton.isEnabled = true
            enhanceButton.isEnabled = true
            progressBar.visibility = ProgressBar.GONE
        }
    }

    override fun onResume() {
        super.onResume()
        registerRestorationReceiver()
    }

    override fun onPause() {
        super.onPause()
        unregisterRestorationReceiver()
    }

    private fun registerRestorationReceiver() {
        android.util.Log.i("MainActivity", "Registering restoration broadcast receiver")
        restorationReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                android.util.Log.i("MainActivity", "Broadcast received: ${intent?.action}")
                when (intent?.action) {
                    RestorationService.BROADCAST_PROGRESS -> {
                        val current = intent.getIntExtra(RestorationService.EXTRA_CURRENT_STEP, 0)
                        val total = intent.getIntExtra(RestorationService.EXTRA_TOTAL_STEPS, 0)
                        updateRestorationProgress(current, total)
                    }
                    RestorationService.BROADCAST_COMPLETE -> {
                        val outputPath = intent.getStringExtra(RestorationService.EXTRA_OUTPUT_PATH)
                        android.util.Log.i("MainActivity", "Restoration complete, output: $outputPath")
                        onRestorationComplete(outputPath)
                    }
                    RestorationService.BROADCAST_ERROR -> {
                        val error = intent.getStringExtra(RestorationService.EXTRA_ERROR_MESSAGE)
                        android.util.Log.e("MainActivity", "Restoration error: $error")
                        onRestorationError(error)
                    }
                }
            }
        }

        val filter = IntentFilter().apply {
            addAction(RestorationService.BROADCAST_PROGRESS)
            addAction(RestorationService.BROADCAST_COMPLETE)
            addAction(RestorationService.BROADCAST_ERROR)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(restorationReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(restorationReceiver, filter)
        }
    }

    private fun unregisterRestorationReceiver() {
        restorationReceiver?.let {
            try {
                unregisterReceiver(it)
            } catch (e: IllegalArgumentException) {
                // Receiver not registered
            }
        }
        restorationReceiver = null
    }

    private fun updateRestorationProgress(current: Int, total: Int) {
        runOnUiThread {
            statusText.text = "Restoring"
        }
    }

    private fun onRestorationComplete(outputPath: String?) {
        android.util.Log.i("MainActivity", "onRestorationComplete called with path: $outputPath")
        runOnUiThread {
            progressBar.visibility = ProgressBar.GONE
            performanceText.visibility = TextView.GONE

            if (outputPath != null) {
                try {
                    android.util.Log.i("MainActivity", "Loading result bitmap from: $outputPath")
                    val bitmap = android.graphics.BitmapFactory.decodeFile(outputPath)
                    if (bitmap != null) {
                        android.util.Log.i("MainActivity", "Bitmap loaded: ${bitmap.width}x${bitmap.height}")
                        currentOutputBitmap = bitmap
                        comparisonSlider.setAfterImage(bitmap)
                        comparisonSlider.reset()

                        statusText.text = "Restoration complete ✓"
                        sliderInstructions.visibility = TextView.VISIBLE
                        saveButton.isEnabled = true

                        // Delete temp file
                        java.io.File(outputPath).delete()
                        android.util.Log.i("MainActivity", "✓ Restoration display complete")
                    } else {
                        android.util.Log.e("MainActivity", "Failed to decode bitmap from file")
                        Toast.makeText(this, "Failed to load result", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    android.util.Log.e("MainActivity", "Error loading result", e)
                    Toast.makeText(this, "Failed to load result: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } else {
                android.util.Log.e("MainActivity", "Output path is null")
            }

            restoreButton.isEnabled = true
            enhanceButton.isEnabled = true
            selectImageButton.isEnabled = true
        }
    }

    private fun onRestorationError(error: String?) {
        runOnUiThread {
            progressBar.visibility = ProgressBar.GONE
            performanceText.visibility = TextView.GONE

            statusText.text = "Error: $error"
            Toast.makeText(this, "Error: $error", Toast.LENGTH_SHORT).show()

            restoreButton.isEnabled = true
            enhanceButton.isEnabled = true
            selectImageButton.isEnabled = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        // Unregister receiver
        unregisterRestorationReceiver()

        // Close Zero-DCE model (restoration models are managed by service)
        zeroDCE?.close()
    }
}
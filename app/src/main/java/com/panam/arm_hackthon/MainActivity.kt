package com.panam.arm_hackthon

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Build
import android.provider.MediaStore
import android.widget.*
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

    private var progressDialog: AlertDialog? = null
    private var restorationReceiver: BroadcastReceiver? = null

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
            currentInputBitmap = bitmap

            // Set before image in slider
            comparisonSlider.setBeforeImage(bitmap)
            comparisonSlider.setAfterImage(bitmap) // Show same image initially
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

                    statusText.text = "Successful"
                    sliderInstructions.visibility = TextView.VISIBLE
                    performanceText.visibility = TextView.VISIBLE
                    performanceText.text = buildString {
                        append("⚡ ${inferenceTime}ms\n")
                        append("📱 ${enhanced.width}x${enhanced.height}\n")
                        append("🚀 Arm NNAPI")
                    }

                    saveButton.isEnabled = true

                    Toast.makeText(
                        this@MainActivity,
                        "Enhanced in ${inferenceTime}ms",
                        Toast.LENGTH_SHORT
                    ).show()
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

            // Show progress dialog
            val progressMessage = TextView(this@MainActivity).apply {
                text = "Restoring: Step 0/100\nThis will take ~10 minutes...\nYou can minimize the app."
                setPadding(60, 40, 60, 40)
                textSize = 16f
            }

            progressDialog = AlertDialog.Builder(this@MainActivity)
                .setTitle("Image Restoration")
                .setView(progressMessage)
                .setCancelable(false)
                .create()
            progressDialog?.show()

            // Disable buttons
            restoreButton.isEnabled = false
            enhanceButton.isEnabled = false
            progressBar.visibility = ProgressBar.VISIBLE
            statusText.text = "Starting restoration service..."

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
            statusText.text = "Restoring: Step $current/$total"
            progressDialog?.findViewById<TextView>(android.R.id.message)?.text =
                "Restoring: Step $current/$total\nPlease wait..."
        }
    }

    private fun onRestorationComplete(outputPath: String?) {
        android.util.Log.i("MainActivity", "onRestorationComplete called with path: $outputPath")
        runOnUiThread {
            android.util.Log.i("MainActivity", "Dismissing progress dialog")
            progressDialog?.dismiss()
            progressDialog = null
            progressBar.visibility = ProgressBar.GONE

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

                        Toast.makeText(this, "Restoration completed!", Toast.LENGTH_SHORT).show()

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
        }
    }

    private fun onRestorationError(error: String?) {
        runOnUiThread {
            progressDialog?.dismiss()
            progressDialog = null
            progressBar.visibility = ProgressBar.GONE

            statusText.text = "Error"
            Toast.makeText(this, "Error: $error", Toast.LENGTH_SHORT).show()

            restoreButton.isEnabled = true
            enhanceButton.isEnabled = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        // Unregister receiver
        unregisterRestorationReceiver()

        // Dismiss progress dialog if showing
        progressDialog?.dismiss()
        progressDialog = null

        // Close Zero-DCE model (restoration models are managed by service)
        zeroDCE?.close()
    }
}
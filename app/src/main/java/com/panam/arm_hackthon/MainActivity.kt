package com.panam.arm_hackthon

import android.content.ContentValues
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.panam.arm_hackthon.ml.ZeroDCE
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var comparisonSlider: com.panam.arm_hackthon.views.ImageComparisonSlider
    private lateinit var selectImageButton: Button
    private lateinit var enhanceButton: Button
    private lateinit var saveButton: Button
    private lateinit var statusText: TextView
    private lateinit var performanceText: TextView
    private lateinit var sliderInstructions: TextView
    private lateinit var progressBar: ProgressBar

    private var zeroDCE: ZeroDCE? = null
    private var currentInputBitmap: Bitmap? = null
    private var currentOutputBitmap: Bitmap? = null

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
        saveButton = findViewById(R.id.saveButton)
        statusText = findViewById(R.id.statusText)
        performanceText = findViewById(R.id.performanceText)
        sliderInstructions = findViewById(R.id.sliderInstructions)
        progressBar = findViewById(R.id.progressBar)

        // Initialize Zero-DCE model
        zeroDCE = ZeroDCE(this)

        lifecycleScope.launch {
            statusText.text = "Initializing..."
            val success = zeroDCE?.initialize(useNNAPI = true) ?: false

            if (success) {
                statusText.text = "Ready"
                enhanceButton.isEnabled = true
            } else {
                statusText.text = "Failed"
                enhanceButton.isEnabled = false
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

        saveButton.setOnClickListener {
            currentOutputBitmap?.let { bitmap ->
                saveImageToGallery(bitmap)
            }
        }

        // Initially disable buttons
        enhanceButton.isEnabled = false
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
            saveButton.isEnabled = false
            sliderInstructions.visibility = TextView.GONE
            statusText.text = "Ready"
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

    override fun onDestroy() {
        super.onDestroy()
        zeroDCE?.close()
    }
}
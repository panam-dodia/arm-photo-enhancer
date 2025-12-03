package com.panam.arm_hackthon.views

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Image Comparison Slider
 * Shows before/after images with draggable divider
 */
class ImageComparisonSlider @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var beforeBitmap: Bitmap? = null
    private var afterBitmap: Bitmap? = null

    private var sliderPosition = 0.5f // 0.0 to 1.0
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 4f
        style = Paint.Style.STROKE
        setShadowLayer(8f, 0f, 0f, Color.BLACK)
    }

    private val handlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
        setShadowLayer(8f, 0f, 0f, Color.BLACK)
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 40f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    private val handleRadius = 40f
    private val arrowSize = 20f

    fun setBeforeImage(bitmap: Bitmap?) {
        beforeBitmap = bitmap
        invalidate()
    }

    fun setAfterImage(bitmap: Bitmap?) {
        afterBitmap = bitmap
        invalidate()
    }

    fun reset() {
        sliderPosition = 0.5f
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val dividerX = viewWidth * sliderPosition

        // Draw black background
        canvas.drawColor(Color.BLACK)

        // Calculate scaled image bounds (FIT_CENTER - preserve aspect ratio)
        val bitmap = beforeBitmap ?: afterBitmap
        if (bitmap != null) {
            val scale = min(viewWidth / bitmap.width, viewHeight / bitmap.height)
            val scaledWidth = bitmap.width * scale
            val scaledHeight = bitmap.height * scale
            val left = (viewWidth - scaledWidth) / 2
            val top = (viewHeight - scaledHeight) / 2
            val right = left + scaledWidth
            val bottom = top + scaledHeight

            // Draw before image (left side)
            beforeBitmap?.let { bmp ->
                // Calculate how much of the source image to show (based on divider position)
                val srcWidth = bmp.width * sliderPosition
                val srcRect = RectF(0f, 0f, srcWidth, bmp.height.toFloat())

                // Calculate destination rect (scaled and positioned)
                val dstWidth = scaledWidth * sliderPosition
                val dstRect = RectF(left, top, left + dstWidth, bottom)

                // Clip to divider
                canvas.save()
                canvas.clipRect(0f, 0f, dividerX, viewHeight)
                canvas.drawBitmap(bmp, srcRect.toRect(), dstRect.toRect(), paint)
                canvas.restore()

                // Draw "BEFORE" label on left side
                if (sliderPosition > 0.2f) {
                    canvas.drawText("BEFORE", dividerX / 2, 60f, textPaint)
                }
            }

            // Draw after image (right side)
            afterBitmap?.let { bmp ->
                // Calculate how much of the source image to show
                val srcLeft = bmp.width * sliderPosition
                val srcRect = RectF(srcLeft, 0f, bmp.width.toFloat(), bmp.height.toFloat())

                // Calculate destination rect
                val dstLeft = left + (scaledWidth * sliderPosition)
                val dstRect = RectF(dstLeft, top, right, bottom)

                // Clip to divider
                canvas.save()
                canvas.clipRect(dividerX, 0f, viewWidth, viewHeight)
                canvas.drawBitmap(bmp, srcRect.toRect(), dstRect.toRect(), paint)
                canvas.restore()

                // Draw "AFTER" label on right side
                if (sliderPosition < 0.8f) {
                    canvas.drawText("AFTER", dividerX + (viewWidth - dividerX) / 2, 60f, textPaint)
                }
            }
        }

        // Draw vertical divider line
        canvas.drawLine(dividerX, 0f, dividerX, viewHeight, dividerPaint)

        // Draw handle (circle with arrows)
        val handleY = viewHeight / 2

        // Circle
        canvas.drawCircle(dividerX, handleY, handleRadius, handlePaint)

        // Left arrow
        val arrowPath = Path().apply {
            moveTo(dividerX - arrowSize, handleY)
            lineTo(dividerX - arrowSize / 2, handleY - arrowSize / 2)
            lineTo(dividerX - arrowSize / 2, handleY + arrowSize / 2)
            close()
        }
        canvas.drawPath(arrowPath, Paint().apply {
            color = Color.parseColor("#4CAF50")
            style = Paint.Style.FILL
        })

        // Right arrow
        val arrowPath2 = Path().apply {
            moveTo(dividerX + arrowSize, handleY)
            lineTo(dividerX + arrowSize / 2, handleY - arrowSize / 2)
            lineTo(dividerX + arrowSize / 2, handleY + arrowSize / 2)
            close()
        }
        canvas.drawPath(arrowPath2, Paint().apply {
            color = Color.parseColor("#4CAF50")
            style = Paint.Style.FILL
        })
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                // Request parent to not intercept touch events
                parent?.requestDisallowInterceptTouchEvent(true)
                sliderPosition = (event.x / width).coerceIn(0f, 1f)
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                // Update slider position based on touch
                sliderPosition = (event.x / width).coerceIn(0f, 1f)
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP,
            MotionEvent.ACTION_CANCEL -> {
                // Allow parent to intercept touch events again
                parent?.requestDisallowInterceptTouchEvent(false)
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    private fun RectF.toRect(): Rect {
        return Rect(left.toInt(), top.toInt(), right.toInt(), bottom.toInt())
    }
}
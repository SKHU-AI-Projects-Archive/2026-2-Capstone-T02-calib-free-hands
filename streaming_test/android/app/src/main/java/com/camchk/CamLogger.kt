package com.camchk

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Logs camera characteristics once and the actual CaptureResult every N frames (logcat + CSV). */
class CamLogger(context: Context, private val everyNFrames: Int) {
    companion object {
        const val TAG = "CAMCHK"
        private const val HEADER =
            "frame,sensor_timestamp_ns,ae_mode,ae_lock,exposure_time_ns,iso,af_mode,focus_distance,eis_mode,ois_mode,ae_state,af_state"
    }

    val file: File
    private val writer: BufferedWriter
    private var count = 0L
    private var closed = false

    init {
        require(everyNFrames > 0) { "log.every_n_frames must be > 0" }
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        file = File(dir, "camchk_$stamp.csv")
        writer = file.bufferedWriter()
        writer.appendLine(HEADER)
        writer.flush()
    }

    fun logCharacteristics(cameraId: String, c: CameraCharacteristics) {
        fun <T> key(k: CameraCharacteristics.Key<T>): String =
            when (val v = c.get(k)) {
                null -> "n/a"
                is IntArray -> v.joinToString(prefix = "[", postfix = "]")
                is FloatArray -> v.joinToString(prefix = "[", postfix = "]")
                is BooleanArray -> v.joinToString(prefix = "[", postfix = "]")
                else -> v.toString()
            }
        Log.i(TAG, "camera id=$cameraId")
        Log.i(TAG, "hardware_level=${key(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)} (0=LIMITED 1=FULL 2=LEGACY 3=LEVEL_3 4=EXTERNAL)")
        Log.i(TAG, "capabilities=${key(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)} (1=MANUAL_SENSOR)")
        Log.i(TAG, "eis_modes=${key(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)} (0=OFF 1=ON 2=PREVIEW_STABILIZATION)")
        Log.i(TAG, "ois_modes=${key(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)} (0=OFF 1=ON)")
        Log.i(TAG, "ae_modes=${key(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES)} (0=OFF 1=ON) ae_lock_available=${key(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE)}")
        Log.i(TAG, "exposure_time_range_ns=${key(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)} iso_range=${key(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)}")
        Log.i(TAG, "af_modes=${key(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)} (0=OFF 1=AUTO 2=MACRO 3=CONTINUOUS_VIDEO 4=CONTINUOUS_PICTURE 5=EDOF)")
        Log.i(TAG, "min_focus_distance_diopter=${key(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)}")
    }

    @Synchronized
    fun onResult(r: TotalCaptureResult) {
        if (closed) return
        val frame = count++
        if (frame % everyNFrames != 0L) return
        val row = listOf(
            frame,
            r.get(CaptureResult.SENSOR_TIMESTAMP),
            r.get(CaptureResult.CONTROL_AE_MODE),
            r.get(CaptureResult.CONTROL_AE_LOCK),
            r.get(CaptureResult.SENSOR_EXPOSURE_TIME),
            r.get(CaptureResult.SENSOR_SENSITIVITY),
            r.get(CaptureResult.CONTROL_AF_MODE),
            r.get(CaptureResult.LENS_FOCUS_DISTANCE),
            r.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE),
            r.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE),
            r.get(CaptureResult.CONTROL_AE_STATE),
            r.get(CaptureResult.CONTROL_AF_STATE),
        ).map { it?.toString() ?: "" }
        Log.i(TAG, HEADER.split(",").zip(row).joinToString(" ") { (k, v) -> "$k=$v" })
        writer.appendLine(row.joinToString(","))
        writer.flush()
    }

    @Synchronized
    fun close() {
        closed = true
        writer.close()
    }
}

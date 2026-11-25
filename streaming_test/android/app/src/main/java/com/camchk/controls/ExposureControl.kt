package com.camchk.controls

import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import com.camchk.CamLogger
import com.camchk.LockSettings
import com.camchk.Section
import com.camchk.bool
import com.camchk.int
import com.camchk.long
import com.camchk.str

class ExposureControl(cfg: Section, private val lockSettings: LockSettings) : CameraControl {
    override val enabled = cfg.bool("enabled")
    private val mode = cfg.str("mode")
    private val exposureTimeNs = cfg.long("exposure_time_ns")
    private val iso = cfg.int("iso")

    // lock mode: AE runs (AE_LOCK=false) until locked = true.
    @Volatile private var locked = mode == "lock" && !lockSettings.waitForConvergence
    private var frames = 0

    // auto mode + long-press: AE_LOCK freezes the current exposure.
    @Volatile private var held = false

    init {
        require(mode in listOf("auto", "lock", "manual")) { "exposure.mode must be auto|lock|manual, got '$mode'" }
    }

    override fun apply(builder: CaptureRequest.Builder) {
        when (mode) {
            "auto" -> {
                builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
                builder.set(CaptureRequest.CONTROL_AE_LOCK, held)
            }
            "lock" -> {
                builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
                builder.set(CaptureRequest.CONTROL_AE_LOCK, locked)
            }
            "manual" -> {
                builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                builder.set(CaptureRequest.CONTROL_AE_LOCK, false)
                builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureTimeNs)
                builder.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
            }
        }
    }

    override fun setHold(hold: Boolean): Boolean {
        if (mode != "auto") return false
        held = hold
        Log.i(CamLogger.TAG, "AE hold=$hold")
        return true
    }

    override fun onResult(result: TotalCaptureResult): Boolean {
        if (mode != "lock" || locked) return false
        frames++
        val converged = result.get(CaptureResult.CONTROL_AE_STATE) == CameraMetadata.CONTROL_AE_STATE_CONVERGED
        if (!converged && frames < lockSettings.timeoutFrames) return false
        locked = true
        Log.i(
            CamLogger.TAG,
            "AE lock: reason=${if (converged) "converged" else "timeout"} after_frames=$frames " +
                "exposure_time_ns=${result.get(CaptureResult.SENSOR_EXPOSURE_TIME)} iso=${result.get(CaptureResult.SENSOR_SENSITIVITY)}"
        )
        return true
    }
}

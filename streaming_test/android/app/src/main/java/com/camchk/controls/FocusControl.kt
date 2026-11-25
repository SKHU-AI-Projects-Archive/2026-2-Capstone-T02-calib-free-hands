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
import com.camchk.float
import com.camchk.str

class FocusControl(cfg: Section, private val lockSettings: LockSettings) : CameraControl {
    override val enabled = cfg.bool("enabled")
    private val mode = cfg.str("mode")
    private val focusDistance = cfg.float("focus_distance")

    // lock mode with wait_for_convergence: continuous AF until lockedDistance is read from a result.
    @Volatile private var lockedDistance: Float? = null
    private var settled = !(mode == "lock" && lockSettings.waitForConvergence) // true = nothing left to wait for
    private var frames = 0

    // auto mode + long-press: the lens position seen in the latest result, and the one frozen by a hold.
    @Volatile private var latestDistance: Float? = null
    @Volatile private var heldDistance: Float? = null

    init {
        require(mode in listOf("auto", "lock", "manual")) { "focus.mode must be auto|lock|manual, got '$mode'" }
    }

    override fun apply(builder: CaptureRequest.Builder) {
        when (mode) {
            "auto" -> {
                val held = heldDistance
                if (held != null) {
                    builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                    builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, held)
                } else {
                    builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                }
            }
            "lock" -> {
                val distance = lockedDistance
                if (distance != null) {
                    builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                    builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, distance)
                } else if (lockSettings.waitForConvergence) {
                    builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                } else {
                    // Previous behavior: AUTO without trigger, the lens stays where it is.
                    builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
                }
            }
            "manual" -> {
                builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, focusDistance)
            }
        }
    }

    override fun setHold(hold: Boolean): Boolean {
        if (mode != "auto") return false
        if (!hold) {
            heldDistance = null
            Log.i(CamLogger.TAG, "AF hold=false (continuous video AF)")
            return true
        }
        val distance = latestDistance
        if (distance == null) {
            Log.w(CamLogger.TAG, "AF hold failed: no LENS_FOCUS_DISTANCE in results yet")
            return false
        }
        heldDistance = distance
        Log.i(CamLogger.TAG, "AF hold=true locked_focus_distance=$distance")
        return true
    }

    override fun onResult(result: TotalCaptureResult): Boolean {
        if (mode == "auto") result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let { latestDistance = it }
        if (settled) return false
        frames++
        val afState = result.get(CaptureResult.CONTROL_AF_STATE)
        // In CONTINUOUS_VIDEO the "in focus" state is PASSIVE_FOCUSED; FOCUSED_LOCKED follows a trigger.
        val focused = afState == CameraMetadata.CONTROL_AF_STATE_PASSIVE_FOCUSED ||
            afState == CameraMetadata.CONTROL_AF_STATE_FOCUSED_LOCKED
        if (!focused && frames < lockSettings.timeoutFrames) return false
        val distance = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
        if (distance == null) {
            if (frames >= lockSettings.timeoutFrames) {
                settled = true
                Log.w(CamLogger.TAG, "AF lock failed: no LENS_FOCUS_DISTANCE in result, staying in CONTINUOUS_VIDEO")
            }
            return false // focused but distance missing: retry next frame until timeout
        }
        settled = true
        lockedDistance = distance
        Log.i(
            CamLogger.TAG,
            "AF lock: reason=${if (focused) "focused" else "timeout"} after_frames=$frames locked_focus_distance=$distance (af_state=$afState)"
        )
        return true
    }
}

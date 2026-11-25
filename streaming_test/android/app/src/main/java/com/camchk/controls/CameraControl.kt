package com.camchk.controls

import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult

interface CameraControl {
    /** false -> the module is skipped and library default behavior is kept. */
    val enabled: Boolean

    /** Writes this control's keys into the running repeating request. */
    fun apply(builder: CaptureRequest.Builder)

    /**
     * Called for every CaptureResult (camera thread).
     * Return true once when apply() would now write different values, so the request is re-submitted.
     */
    fun onResult(result: TotalCaptureResult): Boolean = false

    /**
     * User long-press: hold = true freezes the current auto values (like AE/AF lock in a camera app),
     * hold = false returns to auto. Only meaningful for modules whose mode is "auto".
     * Returns true if the hold state was applied; the caller then re-submits the request with apply().
     */
    fun setHold(hold: Boolean): Boolean = false
}

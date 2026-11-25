package com.camchk.controls

import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import com.camchk.Section
import com.camchk.bool
import com.camchk.onOff

class StabilizationControl(cfg: Section) : CameraControl {
    override val enabled = cfg.bool("enabled")
    private val eis = cfg.onOff("eis")
    private val ois = cfg.onOff("ois")

    override fun apply(builder: CaptureRequest.Builder) {
        builder.set(
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
            if (eis) CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_ON
            else CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF
        )
        builder.set(
            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
            if (ois) CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON
            else CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF
        )
    }
}

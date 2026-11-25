package com.camchk

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraManager
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.util.Log
import android.view.SurfaceView
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import com.camchk.controls.CameraControl
import com.camchk.controls.ExposureControl
import com.camchk.controls.FocusControl
import com.camchk.controls.StabilizationControl
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.codescanner.GmsBarcodeScannerOptions
import com.google.mlkit.vision.codescanner.GmsBarcodeScanning
import com.pedro.common.ConnectChecker
import com.pedro.encoder.input.sources.audio.NoAudioSource
import com.pedro.encoder.input.sources.video.Camera2Source
import com.pedro.library.generic.GenericStream

class MainActivity : Activity(), ConnectChecker {

    // ACCESS_LOCAL_NETWORK exists only on Android 17+ (API 37); without it sends to private addresses
    // (LAN, emulator 10.0.2.2) fail with EPERM. Older versions must not request it.
    private val permissions = buildList {
        add(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 37) add(Manifest.permission.ACCESS_LOCAL_NETWORK)
    }.toTypedArray()
    private val main = Handler(Looper.getMainLooper())

    private lateinit var config: Section
    private lateinit var surfaceView: SurfaceView
    private lateinit var button: Button
    private lateinit var scanButton: Button
    private lateinit var statusView: TextView
    private lateinit var statusCard: View
    private lateinit var streamDot: View
    private lateinit var streamValue: TextView
    private lateinit var lockDot: View
    private lateinit var lockValue: TextView
    private lateinit var elapsedView: TextView
    private lateinit var lockBadge: View
    private lateinit var hintView: TextView
    private lateinit var subtitleView: TextView

    private var stream: GenericStream? = null
    private var logger: CamLogger? = null
    private var lastSession: CameraCaptureSession? = null

    // Kept for the long-press AE/AF hold (set in startCamera).
    private var camera: Camera2Source? = null
    private var activeControls: List<CameraControl> = emptyList()
    private var held = false
    private var heldByPc = false // the current lock was made by the PC (recording), not by long-press
    private var remote: RemoteControl? = null

    // Screen state. Labels and colors match the PC recorder window (pc/recorder_gui.py).
    private enum class StreamUi(val label: String, val color: Int) {
        PREPARING("PREPARING", R.color.idle),
        READY("READY", R.color.idle),
        CONNECTING("CONNECTING", R.color.warn),
        STREAMING("STREAMING", R.color.ok),
        FAILED("FAILED", R.color.rec),
    }
    private var streamUi = StreamUi.PREPARING
    private var streamStartedAt = 0L // SystemClock.elapsedRealtime() when the connection succeeded
    @Volatile private var streamUrl = ""
    private var holdable = false // an exposure/focus module is enabled with mode: auto, so long-press can hold it
    private val clock = object : Runnable {
        override fun run() {
            updateElapsed()
            if (streamUi == StreamUi.STREAMING) main.postDelayed(this, 500)
        }
    }

    // Last connection failure; kept on screen when the stop that follows it reports onDisconnect.
    @Volatile private var failure: String? = null

    // QR: a scanned URL is saved and replaces stream.url from the config until the next scan.
    private val prefs by lazy { getSharedPreferences("camchk", MODE_PRIVATE) }
    private var scanning = false // the scanner (a Play services screen) is open
    private var previewPausedForScan = false
    private var scanNote = false // keep the "scanned ... saved" message until the stream is started
    private var autoScan = true // open the QR scanner when the app starts (once)
    private var startAfterScan = false // a QR code was accepted: start streaming once the preview is back
    private var visible = false // between onResume and onPause

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        surfaceView = findViewById(R.id.preview)
        button = findViewById(R.id.startStop)
        scanButton = findViewById(R.id.scanQr)
        statusView = findViewById(R.id.status)
        statusCard = findViewById(R.id.statusCard)
        streamDot = findViewById(R.id.streamDot)
        streamValue = findViewById(R.id.streamValue)
        lockDot = findViewById(R.id.lockDot)
        lockValue = findViewById(R.id.lockValue)
        elapsedView = findViewById(R.id.elapsed)
        lockBadge = findViewById(R.id.lockBadge)
        hintView = findViewById(R.id.hint)
        subtitleView = findViewById(R.id.subtitle)
        button.isEnabled = false
        button.setOnClickListener { toggleStream() }
        Ui.secondaryButton(scanButton)
        scanButton.setOnClickListener { scanQr() }
        surfaceView.setOnLongClickListener { toggleHold() }
        setStreamUi(StreamUi.PREPARING)
        renderLock()

        try {
            config = Config.load(this)
            holdable = listOf("exposure", "focus").any {
                val sec = config.section(it)
                sec.bool("enabled") && sec.str("mode") == "auto"
            }
        } catch (e: Exception) {
            Log.e(CamLogger.TAG, "config error", e)
            setStatus("The app settings could not be read.")
            setStreamUi(StreamUi.FAILED)
            return
        }
        renderLock()
        renderTarget()
        hintView.animate().alpha(0f).setStartDelay(8000).setDuration(600).start()
        if (permissions.all { checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED }) startCamera()
        else requestPermissions(permissions, 0)
    }

    override fun onRequestPermissionsResult(requestCode: Int, p: Array<out String>, results: IntArray) {
        if (results.isNotEmpty() && results.all { it == PackageManager.PERMISSION_GRANTED }) startCamera()
        else {
            setStatus("Allow camera and local network access to use the app.")
            setStreamUi(StreamUi.FAILED)
        }
    }

    private fun startCamera() {
        try {
            val s = config.section("stream")
            val lock = LockSettings(config.section("lock"))
            val controls = listOf(
                ExposureControl(config.section("exposure"), lock),
                FocusControl(config.section("focus"), lock),
                StabilizationControl(config.section("stabilization")),
            ).filter { it.enabled }
            val camLogger = CamLogger(this, config.section("log").int("every_n_frames"))
            logger = camLogger

            val camera = Camera2Source(this)
            this.camera = camera
            activeControls = controls
            // Must be set before the camera starts: the library only registers its result
            // callback on the repeating request if a custom callback already exists.
            camera.setCustomOnCaptureCompletedCallback { session, _, result ->
                if (session !== lastSession) {
                    // New capture session = camera is running, setCustomRequest works now.
                    lastSession = session
                    main.post { onCameraStarted(camera, controls, camLogger) }
                }
                camLogger.onResult(result)
                // A control returns true when its request values changed (e.g. lock reached): re-apply all.
                if (controls.map { it.onResult(result) }.any { it }) {
                    main.post { camera.setCustomRequest { b -> controls.forEach { it.apply(b) } } }
                }
            }

            // Video only: no microphone, and setOnlyVideo keeps any audio out of the stream and the recording.
            // The library still wants an audio encoder prepared before it starts; it is fed by NoAudioSource.
            val gs = GenericStream(this, this, camera, NoAudioSource())
            gs.getStreamClient().setOnlyVideo(true)
            gs.getGlInterface().autoHandleOrientation = true
            val prepared = gs.prepareVideo(
                s.int("width"), s.int("height"), s.int("bitrate_kbps") * 1000, s.int("fps")
            ) && gs.prepareAudio(44100, true, 64_000)
            if (!prepared) {
                Log.e(CamLogger.TAG, "video configuration failed")
                setStatus("The camera could not be prepared.")
                setStreamUi(StreamUi.FAILED)
                return
            }
            enableServerSilenceCheck(gs)
            stream = gs
            button.isEnabled = true
            setStreamUi(StreamUi.READY)
            if (autoScan) {
                // App start: the QR scanner opens first. It needs the camera, so the preview starts after it.
                autoScan = false
                setStatus("Scan the QR code shown on the PC.")
                scanQr()
            } else {
                gs.startPreview(surfaceView, true)
                setStatus("Camera ready.")
            }
        } catch (e: Exception) {
            Log.e(CamLogger.TAG, "start failed", e)
            setStatus("The camera could not be started.")
            setStreamUi(StreamUi.FAILED)
        }
    }

    // If the PC receiver disappears (program closed, PC off, Wi-Fi gone) the library reports onConnectionFailed
    // after SILENCE_TIMEOUT_MS without any packet from the PC. It is off by default and not exposed by
    // GenericStreamClient, so it is set on the SrtClient through reflection. (setCheckServerAlive would not do:
    // it pings with ICMP, which the PC firewall blocks.)
    private fun enableServerSilenceCheck(gs: GenericStream) {
        try {
            fun Any.field(name: String): Any {
                val f = javaClass.getDeclaredField(name)
                f.isAccessible = true
                return f.get(this)!!
            }
            val srtClient = gs.getStreamClient().field("srtClient").field("srtClient")
            srtClient.javaClass.getMethod("setServerSilenceTimeout", Long::class.javaPrimitiveType)
                .invoke(srtClient, SILENCE_TIMEOUT_MS)
            Log.i(CamLogger.TAG, "server silence timeout=${SILENCE_TIMEOUT_MS}ms")
        } catch (e: Exception) {
            Log.e(CamLogger.TAG, "server silence check not available", e)
        }
    }

    private fun onCameraStarted(camera: Camera2Source, controls: List<CameraControl>, camLogger: CamLogger) {
        val id = camera.getCurrentCameraId()
        val manager = getSystemService(CAMERA_SERVICE) as CameraManager
        camLogger.logCharacteristics(id, manager.getCameraCharacteristics(id))
        if (controls.isEmpty()) {
            Log.i(CamLogger.TAG, "no control enabled: library defaults")
            return
        }
        val ok = camera.setCustomRequest { builder -> controls.forEach { it.apply(builder) } }
        Log.i(CamLogger.TAG, "controls applied=$ok (${controls.joinToString { it.javaClass.simpleName }})")
        if (!scanNote) setStatus("Camera ready.")
    }

    // Long-press on the preview: freeze the current auto exposure/focus, long-press again to go back to auto.
    private fun toggleHold(): Boolean {
        heldByPc = false // the user took over: the PC must not release this lock any more
        setHold(!held, byPc = false)
        return true
    }

    // Lock / release exposure and focus. byPc: the PC asked for it (it locks when a recording starts).
    // Returns false when there was nothing to change.
    private fun setHold(next: Boolean, byPc: Boolean): Boolean {
        val cam = camera ?: return false
        val applied = activeControls.filter { it.setHold(next) }
        if (applied.isEmpty()) {
            if (next && !byPc) setStatus("Nothing to lock with the current settings.")
            return false
        }
        held = next
        renderLock()
        val names = applied.joinToString { it.javaClass.simpleName.removeSuffix("Control") }
        val ok = cam.setCustomRequest { b -> activeControls.forEach { it.apply(b) } }
        Log.i(CamLogger.TAG, "hold=$next by=${if (byPc) "pc" else "user"} ($names) applied=$ok")
        setStatus(
            when {
                byPc && next -> "Exposure and focus locked by the PC for the recording."
                byPc -> "Exposure and focus released by the PC."
                next -> "Exposure and focus locked."
                else -> "Back to automatic exposure and focus."
            }
        )
        return true
    }

    // PC request (recording started / stopped), called from the RemoteControl thread. Waits for the UI thread.
    // The PC releases only what it locked itself, so a lock the user made by long-press stays.
    private fun holdFromPc(on: Boolean): Boolean {
        val locked = java.util.concurrent.atomic.AtomicBoolean(held)
        val done = java.util.concurrent.CountDownLatch(1)
        main.post {
            try {
                if (on) {
                    if (!held && setHold(true, byPc = true)) heldByPc = true
                } else if (heldByPc) {
                    heldByPc = false
                    setHold(false, byPc = true)
                }
            } finally {
                locked.set(held)
                done.countDown()
            }
        }
        done.await(2, java.util.concurrent.TimeUnit.SECONDS)
        return locked.get()
    }

    // The PC may talk to the phone only while streaming (started with the stream, stopped with it).
    private fun startRemote(url: String) {
        val host = Uri.parse(url).host ?: return
        remote?.stop()
        remote = RemoteControl(REMOTE_PORT, host, ::holdFromPc) { held }.also { it.start() }
    }

    private fun stopRemote() {
        remote?.stop()
        remote = null
        if (heldByPc) main.post { heldByPc = false; setHold(false, byPc = true) } // the PC is gone: back to auto
    }

    private fun toggleStream() {
        val gs = stream ?: return
        if (!gs.isStreaming) {
            startStreaming(gs, currentUrl())
        } else {
            gs.stopStream()
            stopRemote()
            setStreamUi(StreamUi.READY)
            setStatus("Streaming stopped.")
        }
    }

    private fun startStreaming(gs: GenericStream, url: String) {
        val protocol = config.section("stream").str("protocol")
        if (!url.startsWith("$protocol://", ignoreCase = true)) {
            Log.e(CamLogger.TAG, "url does not match protocol '$protocol': $url")
            setStatus("This PC address cannot be used. Scan the QR code again.")
            return
        }
        failure = null
        scanNote = false
        streamUrl = url
        Log.i(CamLogger.TAG, "start stream url=$url")
        setStatus("Connecting to the PC ...") // before startStream, so it cannot overwrite a fast failure
        setStreamUi(StreamUi.CONNECTING)
        gs.startStream(url)
    }

    // The URL used for streaming: the last scanned QR code, else stream.url from the config.
    private fun currentUrl(): String = prefs.getString(PREF_URL, null) ?: config.section("stream").str("url")

    // Header: stream settings and where the stream goes (scanned or from the config).
    private fun renderTarget() {
        val s = config.section("stream")
        val linked = prefs.getString(PREF_URL, null) != null
        subtitleView.text = "${s.str("protocol").uppercase()}  ${s.int("width")}x${s.int("height")} @ ${s.int("fps")} fps\n" +
            "→ ${if (linked) "PC linked by QR" else "PC from app settings"}" // no addresses on screen
    }

    // "Scan QR": the PC recorder window (Phone QR) shows srt://<pc ip>:<port>?streamid=publish:<path>.
    private fun scanQr() {
        val gs = stream ?: return
        if (gs.isStreaming) return
        // The scanner is a Play services screen that needs the camera: release it first, restart the preview after.
        gs.stopPreview()
        previewPausedForScan = true
        scanning = true
        val options = GmsBarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build()
        GmsBarcodeScanning.getClient(this, options).startScan()
            .addOnSuccessListener { applyScanned(it.rawValue) }
            .addOnCanceledListener { setStatus("Scan cancelled. Tap Scan QR, or Start streaming to use the last PC.") }
            .addOnFailureListener { e ->
                Log.e(CamLogger.TAG, "scan failed", e)
                // First run on a phone: Google downloads the scanner module, which needs internet once.
                setStatus(
                    if (e.message?.contains("download", ignoreCase = true) == true)
                        "The QR scanner is still being prepared. Connect the phone to the internet once, then tap Scan QR."
                    else "The QR scanner could not be opened."
                )
            }
            .addOnCompleteListener {
                scanning = false
                resumeAfterScan()
            }
    }

    private fun applyScanned(raw: String?) {
        val url = raw?.trim().orEmpty()
        val protocol = config.section("stream").str("protocol")
        val host = Uri.parse(url).host
        if (!url.startsWith("$protocol://", ignoreCase = true) || host.isNullOrEmpty()) {
            Log.w(CamLogger.TAG, "QR code is not a $protocol:// address: $url")
            setStatus("This QR code is not for this app. Scan the code shown on the PC.")
            return
        }
        prefs.edit().putString(PREF_URL, url).apply()
        renderTarget()
        Log.i(CamLogger.TAG, "scanned url=$url")
        // A valid QR code connects right away (once the scanner is closed and the preview is back).
        scanNote = true
        startAfterScan = true
        setStatus("QR code scanned. Connecting to the PC ...")
    }

    // Restart the preview once the scanner is gone and this screen is visible.
    private fun resumeAfterScan() {
        if (!visible || scanning) return
        if (previewPausedForScan) {
            previewPausedForScan = false
            stream?.startPreview(surfaceView, true)
        }
        if (startAfterScan) {
            startAfterScan = false
            val gs = stream
            if (gs != null && !gs.isStreaming) startStreaming(gs, currentUrl())
        }
    }

    override fun onResume() {
        super.onResume()
        visible = true
        resumeAfterScan()
    }

    override fun onPause() {
        visible = false
        super.onPause()
    }

    private fun setStatus(text: String) = runOnUiThread {
        statusCard.setBackgroundResource(R.drawable.bg_card)
        statusView.text = text
    }

    // A warning in the message card (red frame, big red title) and a toast: connection failed / lost.
    private fun showWarning(title: String, detail: String) = runOnUiThread {
        statusCard.setBackgroundResource(R.drawable.bg_card_error)
        val text = SpannableString("$title\n$detail")
        text.setSpan(ForegroundColorSpan(getColor(R.color.rec)), 0, title.length, 0)
        text.setSpan(StyleSpan(Typeface.BOLD), 0, title.length, 0)
        text.setSpan(RelativeSizeSpan(1.3f), 0, title.length, 0)
        statusView.text = text
        Toast.makeText(this, title, Toast.LENGTH_LONG).show()
    }


    // Stream chip, timer and start/stop button (any thread).
    private fun setStreamUi(ui: StreamUi) = runOnUiThread {
        val was = streamUi
        streamUi = ui
        Ui.chip(streamDot, streamValue, getColor(ui.color), ui.label)
        // Red "Stop streaming" only while really streaming; while connecting a neutral "Cancel", so a failed
        // attempt never looks like a running stream.
        when (ui) {
            StreamUi.STREAMING -> {
                button.text = "Stop streaming"
                Ui.primaryButton(button, stop = true)
            }
            StreamUi.CONNECTING -> {
                button.text = "Cancel"
                Ui.neutralButton(button)
            }
            else -> {
                button.text = "Start streaming"
                Ui.primaryButton(button, stop = false)
            }
        }
        scanButton.isEnabled = ui == StreamUi.READY || ui == StreamUi.FAILED // not while connecting/streaming
        main.removeCallbacks(clock)
        if (ui == StreamUi.STREAMING) {
            if (was != StreamUi.STREAMING) streamStartedAt = SystemClock.elapsedRealtime()
            main.post(clock)
        }
        updateElapsed()
        elapsedView.visibility = if (ui == StreamUi.FAILED) View.GONE else View.VISIBLE // room for the warning
    }

    private fun updateElapsed() {
        val streaming = streamUi == StreamUi.STREAMING
        val s = if (streaming) (SystemClock.elapsedRealtime() - streamStartedAt) / 1000 else 0L
        elapsedView.text = "%02d:%02d:%02d".format(s / 3600, s % 3600 / 60, s % 60)
        elapsedView.setTextColor(getColor(if (streaming) R.color.ok else R.color.text_dim))
    }

    // Exposure/focus chip and the LOCKED badge over the preview.
    private fun renderLock() {
        val (color, label) = when {
            !holdable -> R.color.idle to "CONFIG"
            held -> R.color.warn to "LOCKED"
            else -> R.color.idle to "AUTO"
        }
        Ui.chip(lockDot, lockValue, getColor(color), label)
        lockBadge.visibility = if (held) View.VISIBLE else View.GONE
        hintView.text = if (holdable) "Long-press the preview to lock exposure / focus"
        else "Exposure and focus are fixed by the app settings."
    }

    private companion object {
        const val PREF_URL = "stream_url"
        const val SILENCE_TIMEOUT_MS = 6000L
        const val REMOTE_PORT = 8898 // pc/run_config_manual.yaml: phone_port
    }

    override fun onDestroy() {
        super.onDestroy()
        main.removeCallbacksAndMessages(null)
        remote?.stop()
        stream?.release()
        logger?.close()
    }

    // Library callbacks (any thread). Each one is logged with the CAMCHK tag; the status text
    // shows the full reason string. Only onConnectionFailed carries a reason in this library version.
    override fun onConnectionStarted(url: String) {
        Log.i(CamLogger.TAG, "onConnectionStarted url=$url")
        setStreamUi(StreamUi.CONNECTING)
        setStatus("Connecting to the PC ...")
    }

    override fun onConnectionSuccess() {
        Log.i(CamLogger.TAG, "onConnectionSuccess")
        startRemote(streamUrl)
        setStreamUi(StreamUi.STREAMING)
        setStatus("Streaming to the PC.")
    }

    override fun onConnectionFailed(reason: String) {
        Log.e(CamLogger.TAG, "onConnectionFailed reason=$reason")
        val text = "connection failed: $reason"
        failure = text
        runOnUiThread {
            val lost = streamUi == StreamUi.STREAMING // it was working and dropped
            stream?.stopStream()
            stopRemote()
            setStreamUi(StreamUi.FAILED)
            // No addresses and no error codes on screen (they are in the log).
            if (lost) {
                showWarning("CONNECTION LOST", "The stream to the PC stopped.")
            } else {
                showWarning(
                    "NOT CONNECTED",
                    "Could not reach the PC.\n" +
                        "• PC receiver running?\n" +
                        "• Same Wi-Fi?\n" +
                        "• Scan the QR again if the PC changed."
                )
            }
        }
    }

    override fun onDisconnect() {
        Log.w(CamLogger.TAG, "onDisconnect")
        stopRemote()
        if (failure == null) { // after a failure the FAILED chip and the warning stay on screen
            setStreamUi(StreamUi.READY)
            setStatus("Disconnected.")
        }
    }

    override fun onAuthError() {
        Log.e(CamLogger.TAG, "onAuthError")
        val text = "auth error"
        failure = text
        setStreamUi(StreamUi.FAILED)
        showWarning("NOT CONNECTED", "The PC did not accept the connection.")
    }

    override fun onAuthSuccess() {
        Log.i(CamLogger.TAG, "onAuthSuccess")
    }
}

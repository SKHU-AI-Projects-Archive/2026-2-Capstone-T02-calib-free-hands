package com.camchk

import android.content.Context
import org.yaml.snakeyaml.Yaml

typealias Section = Map<String, Any?>

object Config {
    private const val FILE = "config_test_v2.yaml"

    fun load(context: Context): Section =
        context.assets.open(FILE).use { Yaml().load<Section>(it) }
}

/** Shared by the exposure and focus lock modes. */
class LockSettings(cfg: Section) {
    /** Lock after this many frames even if AE/AF never converged. */
    val timeoutFrames = cfg.int("lock_timeout_frames")

    /** false -> lock on the first frame (previous behavior). */
    val waitForConvergence = cfg.bool("wait_for_convergence")

    init {
        require(timeoutFrames > 0) { "lock.lock_timeout_frames must be > 0" }
    }
}

fun Section.section(key: String): Section {
    @Suppress("UNCHECKED_CAST")
    return this[key] as? Section ?: throw IllegalArgumentException("config: missing section '$key'")
}

private fun Section.value(key: String): Any =
    this[key] ?: throw IllegalArgumentException("config: missing key '$key'")

fun Section.str(key: String): String = value(key).toString()
fun Section.int(key: String): Int = (value(key) as Number).toInt()
fun Section.long(key: String): Long = (value(key) as Number).toLong()
fun Section.float(key: String): Float = (value(key) as Number).toFloat()
fun Section.bool(key: String): Boolean = value(key) as Boolean

/** YAML 1.1 turns unquoted on/off into Boolean; accept both that and the strings. */
fun Section.onOff(key: String): Boolean = when (val v = value(key)) {
    is Boolean -> v
    else -> v.toString().lowercase() == "on"
}

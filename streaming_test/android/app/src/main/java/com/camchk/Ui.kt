package com.camchk

import android.content.res.ColorStateList
import android.view.View
import android.widget.Button
import android.widget.TextView

/**
 * Small helpers for the screen. Colors and shapes live in res/values/colors.xml and res/drawable/;
 * the PC recorder window (pc/recorder_gui.py) uses the same palette and structure (header, state chips,
 * big timer, big start/stop button, message card), so keep the two in sync.
 *
 * State colors: ok = streaming/start, rec = recording/stop/error, warn = connecting/locked, idle = nothing going on.
 */
object Ui {
    /** Status dot (drawable/dot.xml as background); call again with another color to change it. */
    fun tint(dot: View, color: Int) {
        dot.backgroundTintList = ColorStateList.valueOf(color)
    }

    /** One chip: the dot and the value text (e.g. "STREAMING") share the state color. */
    fun chip(dot: View, value: TextView, color: Int, label: String) {
        tint(dot, color)
        value.text = label
        value.setTextColor(color)
    }

    /** Small outlined secondary button (drawable/btn_secondary.xml); dim text while disabled. */
    fun secondaryButton(button: Button) {
        val c = button.context
        button.setBackgroundResource(R.drawable.btn_secondary)
        button.setTextColor(
            ColorStateList(
                arrayOf(intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
                intArrayOf(c.getColor(R.color.idle), c.getColor(R.color.text))
            )
        )
    }

    /** Big neutral button ("Cancel" while connecting): not green and not red, so it does not look like a running stream. */
    fun neutralButton(button: Button) {
        button.setBackgroundResource(R.drawable.btn_secondary)
        button.setTextColor(button.context.getColor(R.color.text))
    }

    /** Big primary button: green "start" or red "stop"; grey and flat while disabled. */
    fun primaryButton(button: Button, stop: Boolean) {
        val c = button.context
        button.setBackgroundResource(if (stop) R.drawable.btn_stop else R.drawable.btn_start)
        button.setTextColor(
            ColorStateList(
                arrayOf(intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
                intArrayOf(c.getColor(R.color.text_dim), c.getColor(if (stop) R.color.on_rec else R.color.on_accent))
            )
        )
    }
}

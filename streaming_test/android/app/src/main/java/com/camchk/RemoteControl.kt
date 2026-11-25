package com.camchk

import android.util.Log
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket

/**
 * Tiny HTTP server that lets the PC lock / release exposure and focus (the PC does it when a recording
 * starts / stops, see pc/phone_link.py). It runs only while the phone is streaming and only answers the PC
 * the stream goes to (the host of the scanned address); everybody else gets 403.
 *
 *   POST /hold?on=1   lock exposure + focus     -> {"locked":true}
 *   POST /hold?on=0   release (only what the PC locked)
 *   GET  /status                                  -> {"locked":false}
 *
 * [onHold] is called on the server thread and must return whether exposure/focus are locked afterwards.
 */
class RemoteControl(
    private val port: Int,
    private val allowedHost: String,
    private val onHold: (Boolean) -> Boolean,
    private val isLocked: () -> Boolean,
) {
    @Volatile private var server: ServerSocket? = null

    fun start() {
        if (server != null) return
        val s = try {
            ServerSocket(port)
        } catch (e: Exception) {
            Log.e(CamLogger.TAG, "remote control: port $port not available", e)
            return
        }
        server = s
        Thread({ loop(s) }, "remote-control").apply { isDaemon = true }.start()
        Log.i(CamLogger.TAG, "remote control listening on port $port")
    }

    fun stop() {
        server?.close()
        server = null
    }

    private fun loop(s: ServerSocket) {
        val allowed = try {
            InetAddress.getAllByName(allowedHost).mapNotNull { it.hostAddress?.substringBefore('%') }.toSet()
        } catch (e: Exception) {
            emptySet()
        }
        while (!s.isClosed) {
            val c = try { s.accept() } catch (e: Exception) { break }
            try {
                handle(c, allowed)
            } catch (e: Exception) {
                Log.w(CamLogger.TAG, "remote control request failed: ${e.message}")
            } finally {
                try { c.close() } catch (_: Exception) {}
            }
        }
    }

    private fun handle(c: Socket, allowed: Set<String>) {
        c.soTimeout = 2000
        val peer = c.inetAddress.hostAddress?.substringBefore('%')
        val request = c.getInputStream().bufferedReader().readLine() ?: return
        if (peer == null || peer !in allowed) {
            Log.w(CamLogger.TAG, "remote control: refused a request from outside the PC")
            reply(c, 403, "{\"error\":\"forbidden\"}")
            return
        }
        val target = request.split(" ").getOrNull(1).orEmpty()
        when {
            target.startsWith("/hold") -> {
                val locked = onHold(target.contains("on=1"))
                reply(c, 200, "{\"locked\":$locked}")
            }
            target.startsWith("/status") -> reply(c, 200, "{\"locked\":${isLocked()}}")
            else -> reply(c, 404, "{\"error\":\"not found\"}")
        }
    }

    private fun reply(c: Socket, code: Int, body: String) {
        val reason = when (code) { 200 -> "OK"; 403 -> "Forbidden"; else -> "Not Found" }
        val head = "HTTP/1.1 $code $reason\r\nContent-Type: application/json\r\n" +
            "Content-Length: ${body.toByteArray().size}\r\nConnection: close\r\n\r\n"
        c.getOutputStream().apply { write((head + body).toByteArray()); flush() }
    }
}

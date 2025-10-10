/* While this template provides a good starting point for using Wear Compose, you can always
 * take a look at https://github.com/android/wear-os-samples/tree/main/ComposeStarter to find the
 * most up to date changes to the libraries and their usages.
 */

package com.example.invisibilitycloakdata.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.example.invisibilitycloakdata.R
import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

class MainActivity : ComponentActivity(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var linearAccelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var rotationVector: Sensor? = null

    private lateinit var accelFileOutputStream: FileOutputStream
    private lateinit var linearAccelFileOutputStream: FileOutputStream
    private lateinit var gyroFileOutputStream: FileOutputStream
    private lateinit var rotationFileOutputStream: FileOutputStream

    private lateinit var accelPrintWriter: PrintWriter
    private lateinit var linearAccelPrintWriter: PrintWriter
    private lateinit var gyroPrintWriter: PrintWriter
    private lateinit var rotationPrintWriter: PrintWriter

    private var isRecording = false  // Flag to check if currently recording

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        linearAccelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        val startButton: Button = findViewById(R.id.startButton)
        val stopButton: Button = findViewById(R.id.stopButton)

        startButton.setOnClickListener {
            Toast.makeText(this@MainActivity, "Recording started", Toast.LENGTH_SHORT).show()
            startRecording()
        }

        stopButton.setOnClickListener {
            Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
            stopRecording()
        }
    }

    private fun startRecording() {
        if (!isRecording) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME)
            sensorManager.registerListener(this, linearAccelerometer, SensorManager.SENSOR_DELAY_GAME)
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME)
            sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_GAME)
            val accelFile = File(filesDir, "accelerometer_data.txt")
            val linearAccelFile = File(filesDir, "linear_acceleration_data.txt")
            val gyroFile = File(filesDir, "gyroscope_data.txt")
            val rotationFile = File(filesDir, "rotation_data.txt")
            accelFileOutputStream = FileOutputStream(accelFile, true)
            linearAccelFileOutputStream = FileOutputStream(linearAccelFile, true)
            gyroFileOutputStream = FileOutputStream(gyroFile, true)
            rotationFileOutputStream = FileOutputStream(rotationFile, true)
            accelPrintWriter = PrintWriter(accelFileOutputStream)
            linearAccelPrintWriter = PrintWriter(linearAccelFileOutputStream)
            gyroPrintWriter = PrintWriter(gyroFileOutputStream)
            rotationPrintWriter = PrintWriter(rotationFileOutputStream)
            isRecording = true
        }
    }

    private fun stopRecording() {
        if (isRecording) {
            sensorManager.unregisterListener(this)
            accelPrintWriter.close()
            linearAccelPrintWriter.close()
            gyroPrintWriter.close()
            rotationPrintWriter.close()
            accelFileOutputStream.close()
            linearAccelFileOutputStream.close()
            gyroFileOutputStream.close()
            rotationFileOutputStream.close()
            isRecording = false
        }
    }


    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            if (isRecording) {
                // Get the current timestamp in UTC-8
                val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
                    .withZone(ZoneId.of("America/Los_Angeles"))
                val localTime = formatter.format(Instant.ofEpochMilli(System.currentTimeMillis()))

                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        val data = "Accel, $localTime, ${event.values.joinToString()}"
                        accelPrintWriter.println(data)
                    }
                    Sensor.TYPE_LINEAR_ACCELERATION -> {
                        val data = "LinearAccel, $localTime, ${event.values.joinToString()}"
                        linearAccelPrintWriter.println(data)
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        val data = "Gyro, $localTime, ${event.values.joinToString()}"
                        gyroPrintWriter.println(data)
                    }
                    Sensor.TYPE_ROTATION_VECTOR -> {
                        val data = "Rotation, $localTime, ${event.values.joinToString()}"
                        rotationPrintWriter.println(data)
                    }
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle sensor accuracy changes if needed
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) {
            sensorManager.unregisterListener(this)
            accelPrintWriter.close()
            linearAccelPrintWriter.close()
            gyroPrintWriter.close()
            rotationPrintWriter.close()
            accelFileOutputStream.close()
            linearAccelFileOutputStream.close()
            gyroFileOutputStream.close()
            rotationFileOutputStream.close()
        }
    }
}
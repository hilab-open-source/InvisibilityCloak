package com.example.demo.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.compose.*
import androidx.wear.compose.material.*
import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.IOException
import java.io.OutputStream
import java.io.PrintWriter
import java.net.Socket
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import com.example.demo.R


class MainActivity : ComponentActivity(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var sensorSocket: Socket? = null
    private var sensorOutputStream: OutputStream? = null
    private var isRecording by mutableStateOf(false)

    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("MainActivity", "Launching App")

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        setContent {
            val navController = rememberNavController()
            NavHost(navController = navController, startDestination = "imu_screen") {
                composable("imu_screen") {
                    IMUScreen(
                        onStartRecording = { startRecording() },
                        onStopRecording = { stopRecording() },
                        navController = navController
                    )
                }
                composable("buttons_screen") {
                    ButtonsScreen(navController)
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        Log.d("IMU", "Sensor accuracy changed: $accuracy")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event != null && isRecording) {
            val sensorType = when (event.sensor.type) {
                Sensor.TYPE_LINEAR_ACCELERATION -> "ACCELEROMETER"
                Sensor.TYPE_GYROSCOPE -> "GYROSCOPE"
                else -> return
            }

            val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
                .withZone(ZoneId.of("America/Los_Angeles"))
            val localTime = formatter.format(Instant.now())

            val jsonData = JSONObject().apply {
                put("type", sensorType)
                put("timestamp", localTime)
                put("x", event.values[0])
                put("y", event.values[1])
                put("z", event.values[2])
            }

            coroutineScope.launch {
                try {
                    sensorOutputStream?.write((jsonData.toString() + "\n").toByteArray())
                    sensorOutputStream?.flush()
//                    Log.d("IMU", "Sent: $jsonData")
                } catch (e: IOException) {
                    Log.e("IMU", "Error sending data", e)
                }
            }
        }
    }

    private fun startRecording() {
        coroutineScope.launch {
            try {
                sensorSocket = Socket("192.168.0.222", SENSOR_PORT).also {
                    sensorOutputStream = BufferedOutputStream(it.getOutputStream())
                }

                sensorManager.registerListener(
                    this@MainActivity,
                    sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION),
                    SensorManager.SENSOR_DELAY_GAME
                )
                sensorManager.registerListener(
                    this@MainActivity,
                    sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
                    SensorManager.SENSOR_DELAY_GAME
                )

                isRecording = true

                // Ensure Toast runs on the Main thread
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Recording started", Toast.LENGTH_LONG).show()
                }

                Log.d("IMU", "Recording started")
            } catch (e: IOException) {
                Log.e("IMU", "Failed to start recording", e)
            }
        }
    }

    private fun stopRecording() {
        coroutineScope.launch {
            try {
                if (isRecording) {
                    sensorManager.unregisterListener(this@MainActivity)
                    sensorOutputStream?.close()
                    sensorSocket?.close()
                    sensorOutputStream = null
                    sensorSocket = null
                    isRecording = false

                    // Ensure Toast runs on the Main thread
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Recording stopped", Toast.LENGTH_LONG).show()
                    }

                    Log.d("IMU", "Recording stopped")
                }
            } catch (e: IOException) {
                Log.e("IMU", "Failed to stop recording", e)
            }
        }
    }

    companion object {
        private const val SENSOR_PORT = 8006
    }
}

@Composable
fun IMUScreen(onStartRecording: () -> Unit, onStopRecording: () -> Unit, navController: NavController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // First row: Start and Stop Chips
        Row(
            horizontalArrangement = Arrangement.SpaceEvenly,
            modifier = Modifier.fillMaxWidth()
        ) {
            WearChip(text = "START", onClick = onStartRecording, modifier = Modifier.weight(1f))
            Spacer(modifier = Modifier.width(6.dp)) // Spacing between buttons
            WearChip(text = "STOP", onClick = onStopRecording, modifier = Modifier.weight(1f))
        }

        Spacer(modifier = Modifier.height(16.dp)) // Space between rows

        // Second row: Request Chip
        WearChip(text = "       REQUEST      ", onClick = { navController.navigate("buttons_screen") })
    }
}


private fun sendMessageToPython(context: Context, message: String) {
    CoroutineScope(Dispatchers.IO).launch {
        try {
            Socket("192.168.0.222", 8840).use { socket ->
                PrintWriter(socket.getOutputStream(), true).apply {
                    println(message)
                    Log.d("PythonComm", "Message sent: $message")
                }
            }

            // Show Toast on the Main thread
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Sent: $message", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            Log.e("PythonComm", "Error sending message", e)
        }
    }
}



@Composable
fun ButtonsScreen(navController: NavController) {
    val context = LocalContext.current  // Get the context for Toast messages

    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        val customBlue = Color(0xFF2196F3)
        val customGreen = Color(0xFF4CAF50)
        val customYellow = Color(0xFFFFEB3B)
        val customRed = Color(0xFFF44336)
        val customPurple = Color(0xFFD0BFFF)

        val buttonSize = 57.dp

        // Top Button (A)
        Button(
            onClick = { sendMessageToPython(context, "SKELETON") },
            colors = ButtonDefaults.buttonColors(backgroundColor = Color.Black),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .size(buttonSize)
        ) {
            Image(
                painter = painterResource(id = R.drawable.skeleton),
                contentDescription = "Skeleton button",
                contentScale = ContentScale.Fit, // Ensure proper scaling
                modifier = Modifier.size(buttonSize) // Adjust image size
            )
        }

        // Right Button (D)
        Button(
            onClick = { sendMessageToPython(context, "ERASE") },
            colors = ButtonDefaults.buttonColors(backgroundColor = Color.Black),
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .size(buttonSize)
        ) {
            Image(
                painter = painterResource(id = R.drawable.mask),
                contentDescription = "Mask button",
                contentScale = ContentScale.Fit, // Ensure proper scaling
                modifier = Modifier.size(buttonSize) // Adjust image size
            )
        }

        // Bottom Button (C)
        Button(
            onClick = { sendMessageToPython(context, "INPAINTING") },
            colors = ButtonDefaults.buttonColors(backgroundColor = Color.Black),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .size(buttonSize)
        ) {
            Image(
                painter = painterResource(id = R.drawable.inpainting),
                contentDescription = "Inpainting button",
                contentScale = ContentScale.Fit, // Ensure proper scaling
                modifier = Modifier.size(buttonSize) // Adjust image size
            )
        }

        // Left Button (B)
        Button(
            onClick = { sendMessageToPython(context, "BLURRY") },
            colors = ButtonDefaults.buttonColors(backgroundColor = Color.Black),
            modifier = Modifier
                .align(Alignment.CenterStart)
                .size(buttonSize)
        ) {
            Image(
                painter = painterResource(id = R.drawable.blurry),
                contentDescription = "Blurry button",
                contentScale = ContentScale.Fit, // Ensure proper scaling
                modifier = Modifier.size(buttonSize) // Adjust image size
            )
        }

        // Center Button (Back)
        Button(
            onClick = { sendMessageToPython(context, "RAW") },
            colors = ButtonDefaults.buttonColors(backgroundColor = Color.Black),
            modifier = Modifier
                .align(Alignment.Center)
                .size(buttonSize)
        ) {
            Image(
                painter = painterResource(id = R.drawable.raw),
                contentDescription = "Raw button",
                contentScale = ContentScale.Fit, // Ensure proper scaling
                modifier = Modifier.size(buttonSize) // Adjust image size
            )
        }
    }
}



@Composable
fun WearChip(text: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Chip(
        onClick = onClick,
        label = {
            Text(
                text = text,
                fontSize = 14.sp,
                color = Color.Black,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
        },
        modifier = modifier.height(46.dp),
        colors = ChipDefaults.primaryChipColors(
            backgroundColor = Color(0xFFD0BFFF), // Light purple background
            contentColor = Color.Black
        )
    )
}
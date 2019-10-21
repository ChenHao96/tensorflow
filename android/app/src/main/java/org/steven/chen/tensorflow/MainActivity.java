package org.steven.chen.tensorflow;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import org.steven.chen.tensorflow.camera.CameraActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        super.setContentView(R.layout.activity_main);
        super.setTitle(R.string.app_name);

        Button openCamera = findViewById(R.id.button_camera);
        openCamera.setOnClickListener(view -> startActivity(
                new Intent(MainActivity.this, CameraActivity.class)));

        Button orientation = findViewById(R.id.button_orientation);
        orientation.setOnClickListener(view -> startActivity(
                new Intent(MainActivity.this, OrientationActivity.class)));
    }
}

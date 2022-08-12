package com.yeyupiaoling.android;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        findViewById(R.id.camera_rec_btn).setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, CameraRecognitionActivity.class);
            startActivity(intent);
        });
        findViewById(R.id.photo_rec_btn).setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, RecognitionPhotoActivity.class);
            startActivity(intent);
        });
    }
}
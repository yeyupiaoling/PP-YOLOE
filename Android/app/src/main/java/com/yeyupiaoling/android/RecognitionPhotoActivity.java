package com.yeyupiaoling.android;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.yeyupiaoling.ai.DetectionPredictor;
import com.yeyupiaoling.ai.bean.RecognitionResult;
import com.yeyupiaoling.ai.utils.Utils;

import java.io.FileInputStream;
import java.util.List;

public class RecognitionPhotoActivity extends AppCompatActivity {
    private static final String TAG = RecognitionPhotoActivity.class.getName();
    // 输入到模型的图片大小
    private static final long[] inputShape = new long[]{1, 3, 320, 320};
    private static final int NUM_THREADS = 4;
    private static final float THRESHOLD = 0.5f;
    private static final String modelFile = "detect_model.nb";
    private static final String labelFile = "label_list.txt";
    private ImageView imageView;
    private DetectionPredictor detectionPredictor;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition_photo);

        if (!hasPermission()) {
            requestPermission();
        }

        // 获取控件
        Button selectImgBtn = findViewById(R.id.select_img_btn);
        imageView = findViewById(R.id.image_view);
        selectImgBtn.setOnClickListener(v -> {
            // 打开相册
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent, 1);
        });

        // 获取目标检测器
        try {
            detectionPredictor = new DetectionPredictor(RecognitionPhotoActivity.this,
                    modelFile, labelFile, inputShape, NUM_THREADS, THRESHOLD);
            Log.d(TAG, "模型加载成功！");
        } catch (Exception e) {
            Log.d(TAG, "模型加载失败！");
            e.printStackTrace();
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        String image_path;
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 1) {
                if (data == null) {
                    Log.w("onActivityResult", "user photo data is null");
                    return;
                }
                Uri image_uri = data.getData();
                image_path = Utils.getPathFromURI(RecognitionPhotoActivity.this, image_uri);
                try {
                    // 预测图像
                    FileInputStream fis = new FileInputStream(image_path);
                    Bitmap rotationBitmap = BitmapFactory.decodeStream(fis);
                    imageView.setImageBitmap(rotationBitmap);
                    long start = System.currentTimeMillis();
                    List<RecognitionResult> results = detectionPredictor.predictImage(rotationBitmap);
                    Bitmap bitmap = DetectionPredictor.draw(rotationBitmap, results);
                    imageView.setImageBitmap(bitmap);
                    long end = System.currentTimeMillis();
                    Log.d("RecognitionPhoto", "总预测时间：" + (end - start) + "ms");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }


    // check had permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    // request permission
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
    }
}
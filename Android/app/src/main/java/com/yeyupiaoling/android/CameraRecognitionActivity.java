package com.yeyupiaoling.android;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.yeyupiaoling.ai.DetectionPredictor;
import com.yeyupiaoling.ai.bean.RecognitionResult;
import com.yeyupiaoling.ai.utils.Utils;
import com.yeyupiaoling.ai.utils.YuvToRgbConverter;
import com.yeyupiaoling.ai.view.CanvasView;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraRecognitionActivity extends AppCompatActivity {
    private static final String TAG = CameraRecognitionActivity.class.getSimpleName();
    // 输入到模型的图片大小
    private static final long[] inputShape = new long[]{1, 3, 320, 320};
    private static final int NUM_THREADS = 4;
    private static final float THRESHOLD = 0.5f;
    private static final String modelFile = "detect_model.nb";
    private static final String labelFile = "label_list.txt";

    private final boolean isInfer = true;
    private PreviewView viewFinder;
    private CanvasView mCanvasView;
    // 使用前摄像头
    private final CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

    private ExecutorService cameraExecutor;
    private Bitmap bitmapBuffer = null;
    private YuvToRgbConverter converter;
    private int lastRotation = 0;
    private ImageView imageView;
    private DetectionPredictor detectionPredictor;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_recognition);

        imageView = findViewById(R.id.image_view);
        viewFinder = findViewById(R.id.viewFinder);
        mCanvasView = findViewById(R.id.canvas_view);
        mCanvasView.populateResultList(null);

        // 请求权限
        if (hasPermission()) {
            startCamera();
        } else {
            requestPermission();
            startCamera();
        }

        cameraExecutor = Executors.newSingleThreadExecutor();

        // 把图像数据转换为RGB格式图像
        converter = new YuvToRgbConverter(this);

        // 获取目标检测器
        try {
            detectionPredictor = new DetectionPredictor(CameraRecognitionActivity.this,
                    modelFile, labelFile, inputShape, NUM_THREADS, THRESHOLD);
            Log.d(TAG, "模型加载成功！");
        } catch (Exception e) {
            Log.d(TAG, "模型加载失败！");
            e.printStackTrace();
        }

        // 长按隐藏结果图片
        imageView.setOnLongClickListener(view -> {
            imageView.setVisibility(View.GONE);
            return true;
        });
    }

    @SuppressLint("UnsafeOptInUsageError")
    private void infer(ImageProxy image) {
        int rotation = image.getImageInfo().getRotationDegrees();
        if (bitmapBuffer == null || lastRotation != rotation) {
            lastRotation = rotation;
            bitmapBuffer = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
        }
        Log.d(TAG, "方向：" + rotation + ", 宽：" + image.getWidth() + ", 高：" + image.getHeight());

        mCanvasView.setSize(image.getHeight(), image.getWidth());

        // 将图像转换为RGB，并将其放在bitmapBuffer
        long start = System.currentTimeMillis();
        converter.yuvToRgb(Objects.requireNonNull(image.getImage()), bitmapBuffer);
        Bitmap rotationBitmap = rotation == 0 ? bitmapBuffer : Utils.adjustRotation(bitmapBuffer, rotation);
        Log.d(TAG, "YUV转Bitmap时间：" + (System.currentTimeMillis() - start) + "ms");
        // 预测
        start = System.currentTimeMillis();
        List<RecognitionResult> results = detectionPredictor.predictImage(rotationBitmap);
        Log.d(TAG, "总预测时间：" + (System.currentTimeMillis() - start) + "ms");
        // 画出结果
        mCanvasView.populateResultList(results);
        if (results.size() > 0) {
            Bitmap bitmap = DetectionPredictor.draw(rotationBitmap, results);
            runOnUiThread(() -> {
                imageView.setImageBitmap(bitmap);
            });
        }
        if (results.size() == 0) {
            runOnUiThread(() -> imageView.setImageBitmap(rotationBitmap));
        }
        image.close();
    }


    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(CameraRecognitionActivity.this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // 设置相机支持预览
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

                // 设置相机支持图像分析
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                Log.d("test", viewFinder.getHeight() + ", " + viewFinder.getWidth());

                // 图像分析监听
                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), image -> {
                    if (isInfer) {
                        new Thread(() -> infer(image)).start();
                    }
                });

                // 在重新绑定之前取消绑定用例
                cameraProvider.unbindAll();

                // 将用例绑定到摄像机
                Camera camera = cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        preview,
                        imageAnalysis);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 关闭相机
        cameraExecutor.shutdown();
    }

    // check had permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    // request permission
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
        }
    }
}
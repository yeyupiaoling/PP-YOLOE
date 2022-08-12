package com.yeyupiaoling.ai;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;
import com.yeyupiaoling.ai.bean.RecognitionResult;
import com.yeyupiaoling.ai.utils.ImageUtil;
import com.yeyupiaoling.ai.utils.Utils;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class DetectionPredictor {
    private static final String TAG = DetectionPredictor.class.getName();
    private static final boolean bgr2rgb = true;
    private static final float inputScale = 255.0f;
    private static final float[] inputMean = new float[]{0.485f, 0.456f, 0.406f};
    private static final float[] inputStd = new float[]{0.229f, 0.224f, 0.225f};
    private static final int NUM_THREADS = 4;
    private static final float THRESHOLD = 0.5f;
    private static volatile DetectionPredictor detectionPredictor;
    private final PaddlePredictor paddlePredictor;
    private final Tensor imageTensor;
    private final Tensor scaleFactorTensor;
    private final float threshold;
    private final long[] inputShape;
    private String[] labels;


    public static DetectionPredictor getInstance(Context context, String modelFile,
                                                 String labelFile, long[] inputShape) {
        if (detectionPredictor == null) {
            synchronized (DetectionPredictor.class) {
                if (detectionPredictor == null) {
                    try {
                        detectionPredictor = new DetectionPredictor(context, modelFile,
                                labelFile, inputShape, NUM_THREADS, THRESHOLD);
                        Log.d(TAG, "模型加载成功！");
                    } catch (Exception e) {
                        Log.d(TAG, "模型加载失败！");
                        e.printStackTrace();
                    }
                }
            }
        }
        return detectionPredictor;
    }


    public DetectionPredictor(Context context, String modelFile, String labelFile, long[] inputShape,
                              int numThreads, float threshold) throws Exception {
        readListFromFile(context, labelFile);
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV初始化成功！");
        } else {
            Log.e(TAG, "OpenCV初始化失败!");
        }
        // 复制模型文件
        String modelPath = context.getCacheDir().getAbsolutePath() + File.separator + modelFile;
        File fileCopy = new File(modelPath);
        Utils.copyFileFromAsset(context, fileCopy.getName(), modelPath);
        // 加载模型
        File file = new File(modelPath);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }
        try {
            MobileConfig config = new MobileConfig();
            config.setModelFromFile(modelPath);
            config.setThreads(numThreads);
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
            paddlePredictor = PaddlePredictor.createPaddlePredictor(config);

            imageTensor = paddlePredictor.getInput(0);
            imageTensor.resize(inputShape);
            scaleFactorTensor = paddlePredictor.getInput(1);
            scaleFactorTensor.resize(new long[]{1, 2});
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("load model fail!");
        }
        this.threshold = threshold;
        this.inputShape = inputShape;
    }


    // 读取类别名称文件
    public void readListFromFile(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();
        ArrayList<String> list = new ArrayList<>();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new InputStreamReader(assetManager.open(filePath)));
            String line;
            while ((line = reader.readLine()) != null) {
                list.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        labels = list.toArray(new String[0]);
    }

    public List<RecognitionResult> predictImage(String imagePath) {
        Log.d(TAG, imagePath);
        assert new File(imagePath).exists() : "image file is not exists!";
        long start = System.currentTimeMillis();
        Mat mat = ImageUtil.read(imagePath);
        Log.d(TAG, "数据读取时间：" + (System.currentTimeMillis() - start) + "ms");
        return predictImage(mat);
    }

    public List<RecognitionResult> predictImage(Bitmap bitmap) {
        long start = System.currentTimeMillis();
        Mat mat = ImageUtil.read(bitmap);
        Log.d(TAG, "数据读取时间：" + (System.currentTimeMillis() - start) + "ms");
        return predictImage(mat);
    }

    public List<RecognitionResult> predictImage(Mat mat) {
        if (bgr2rgb) {
            mat = ImageUtil.bgr2rgb(mat);
        }
        long start = System.currentTimeMillis();
        int width = (int) inputShape[2];
        int height = (int) inputShape[3];
        float[] scaleFactor = new float[]{(float) height / mat.rows(), (float) width / mat.cols()};
        Mat mat1 = ImageUtil.resize(mat, width, height);
        float[] data;
        if (this.inputScale == 1) {
            data = ImageUtil.normalize(mat1, inputMean, inputStd);
        } else {
            data = ImageUtil.normalize(mat1, inputScale, inputMean, inputStd);
        }
        Log.d(TAG, "数据预处理时间：" + (System.currentTimeMillis() - start) + "ms");
        return predict(data, scaleFactor);
    }

    public List<RecognitionResult> predict(float[] inputData, float[] scaleFactor) {
        imageTensor.setData(inputData);
        scaleFactorTensor.setData(scaleFactor);

        try {
            long start = System.currentTimeMillis();
            paddlePredictor.run();
            long end = System.currentTimeMillis();
            Log.d(TAG, "单纯预测时间：" + (end - start) + "ms");
        } catch (Exception e) {
            e.printStackTrace();
        }
        Tensor outputTensor = paddlePredictor.getOutput(0);
        List<RecognitionResult> result = new ArrayList<>();
        float[] output = outputTensor.getFloatData();
        for (int i = 0; i < output.length; i += 6) {
            String label = this.labels[(int) output[i]];
            float score = output[i + 1];
            if (score < this.threshold) continue;
            float left = output[i + 2];
            float top = output[i + 3];
            float right = output[i + 4];
            float bottom = output[i + 5];
            result.add(new RecognitionResult(label, score, left, top, right, bottom));
        }
        return result;
    }


    // 画框和识别名字
    public static Bitmap draw(Bitmap bitmap, List<RecognitionResult> results) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paintRect = new Paint();
        paintRect.setColor(Color.GREEN);
        paintRect.setStyle(Paint.Style.STROKE);
        paintRect.setStrokeWidth(5);
        Paint paintText = new Paint();
        paintText.setColor(Color.GREEN);
        paintText.setStyle(Paint.Style.FILL);
        paintText.setAntiAlias(false);
        paintText.setTextSize(30);
        paintText.setFakeBoldText(true);

        for (int i = 0; i < results.size(); i++) {
            RecognitionResult recognitionResult = results.get(i);
            String label = recognitionResult.label;
            int left = (int) recognitionResult.left;
            int top = (int) recognitionResult.top;
            int right = (int) recognitionResult.right;
            int bottom = (int) recognitionResult.bottom;

            canvas.drawRect(left, top, right, bottom, paintRect);
            canvas.drawText(label, left, top - 5, paintText);
        }
        return mutableBitmap;
    }

}

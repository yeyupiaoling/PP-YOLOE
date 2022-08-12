package com.yeyupiaoling.ai.utils;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ImageUtil {

    // 读取图片成Mat
    public static Mat read(String path) {
        return Imgcodecs.imread(path, Imgcodecs.IMREAD_COLOR);
    }

    // 读取图片成Mat
    public static Mat read(Bitmap bitmap) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat, true);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2BGR);
        return mat;
    }


    // 将mat由bgr转rgb
    public static Mat bgr2rgb(Mat mat) {
        Mat matRGB = new Mat();
        Imgproc.cvtColor(mat, matRGB, Imgproc.COLOR_BGR2RGB);
        return matRGB;
    }

    // 缩放图片
    public static Mat resize(Mat mat, int width, int height) {
        Size size = new Size(width, height);
        Mat mat1 = new Mat(size, CvType.CV_16FC3);
        Imgproc.resize(mat, mat1, size);
        return mat1;
    }

    // Mat转成Bitmap
    public static Bitmap matToBitmap(Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Mat mat1 = new Mat();
        Imgproc.cvtColor(mat, mat1, Imgproc.COLOR_BGR2RGB);
        Utils.matToBitmap(mat1, bitmap);
        return bitmap;
    }

    // 裁剪图片
    public static Mat crop(Mat mat, int left, int top, int right, int bottom){
        int width = right - left;
        int height = bottom - top;
        Rect rect = new Rect(left, top, width, height);
        return new Mat(mat, rect);
    }

    // 归一化
    public static float[] normalize(Mat mat, float[] mean, float[] std) {
        mat.convertTo(mat, CvType.CV_32FC3);
        Mat result = new Mat(mat.rows(), mat.cols(), CvType.CV_32FC3);
        // 执行归一化
        Core.subtract(mat, new Scalar(mean[0], mean[1], mean[2]), result);
        Core.divide(result, new Scalar(std[0], std[1], std[2]), result);
        // 按照bbb...ggg...rrr将数据提取到数组中
        float[] b = new float[(int) result.total()];
        float[] g = new float[(int) result.total()];
        float[] r = new float[(int) result.total()];
        List<Mat> matList = new ArrayList<>();
        Core.split(result, matList);
        matList.get(0).get(0, 0, b);
        matList.get(1).get(0, 0, g);
        matList.get(2).get(0, 0, r);
        float[] data = new float[(int) result.total() * 3];
        System.arraycopy(b, 0, data, 0, b.length);
        System.arraycopy(g, 0, data, b.length, g.length);
        System.arraycopy(r, 0, data, b.length + g.length, r.length);
        return data;
    }


    // 归一化
    public static float[] normalize(Mat mat, float scale, float[] mean, float[] std) {
        mat.convertTo(mat, CvType.CV_32FC3);
        Mat result = new Mat(mat.rows(), mat.cols(), CvType.CV_32FC3);
        // 执行归一化
        Core.divide(mat, new Scalar(scale, scale, scale), result);
        Core.subtract(result, new Scalar(mean[0], mean[1], mean[2]), result);
        Core.divide(result, new Scalar(std[0], std[1], std[2]), result);
        // 按照bbb...ggg...rrr将数据提取到数组中
        float[] b = new float[(int) result.total()];
        float[] g = new float[(int) result.total()];
        float[] r = new float[(int) result.total()];
        List<Mat> matList = new ArrayList<>();
        Core.split(result, matList);
        matList.get(0).get(0, 0, b);
        matList.get(1).get(0, 0, g);
        matList.get(2).get(0, 0, r);
        float[] data = new float[(int) result.total() * 3];
        System.arraycopy(b, 0, data, 0, b.length);
        System.arraycopy(g, 0, data, b.length, g.length);
        System.arraycopy(r, 0, data, b.length + g.length, r.length);
        return data;
    }
}

package com.yeyupiaoling.ai.bean;

import androidx.annotation.NonNull;

public class RecognitionResult {
    public float left;
    public float top;
    public float right;
    public float bottom;
    public float score;
    public String label;

    public RecognitionResult() {
    }

    public RecognitionResult(String label) {
        this.label = label;
    }

    public RecognitionResult(float left, float top, float right, float bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    public RecognitionResult(String label, float score, float left, float top, float right, float bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
        this.score = score;
        this.label = label;
    }

    @NonNull
    @Override
    public String toString() {
        return "RecognitionResult{" +
                "left=" + left +
                ", top=" + top +
                ", right=" + right +
                ", bottom=" + bottom +
                ", label=" + label +
                '}';
    }
}
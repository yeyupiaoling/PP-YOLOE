package com.yeyupiaoling.ai.view;

import android.content.Context;
import android.view.GestureDetector;
import android.view.GestureDetector.SimpleOnGestureListener;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.ScaleGestureDetector.OnScaleGestureListener;
import android.view.ScaleGestureDetector.SimpleOnScaleGestureListener;
import android.view.View;

public class CameraXPreviewViewTouchListener implements View.OnTouchListener {
    private GestureDetector mGestureDetector;
    private CustomTouchListener mCustomTouchListener = null;
    private ScaleGestureDetector mScaleGestureDetector;

    public CameraXPreviewViewTouchListener(Context context) {
        // 缩放监听
        OnScaleGestureListener onScaleGestureListener = new SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(ScaleGestureDetector detector) {
                float delta = detector.getScaleFactor();
                if (mCustomTouchListener != null) {
                    mCustomTouchListener.zoom(delta);
                }
                return true;
            }
        };

        // 点击监听
        SimpleOnGestureListener onGestureListener = new SimpleOnGestureListener() {
            @Override
            public void onLongPress(MotionEvent e) {
                if (mCustomTouchListener != null) {
                    // 长按
                    mCustomTouchListener.longPress(e.getX(), e.getY());
                }
            }

            @Override
            public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
                return true;
            }

            @Override
            public boolean onSingleTapConfirmed(MotionEvent e) {
                if (mCustomTouchListener != null) {
                    // 单击
                    mCustomTouchListener.click(e.getX(), e.getY());
                }
                return true;
            }

            @Override
            public boolean onDoubleTap(MotionEvent e) {
                if (mCustomTouchListener != null) {
                    // 双击
                    mCustomTouchListener.doubleClick(e.getX(), e.getY());
                }
                return true;
            }
        };

        mGestureDetector = new GestureDetector(context, onGestureListener);
        mScaleGestureDetector = new ScaleGestureDetector(context, onScaleGestureListener);
    }

    public void setCustomTouchListener(CustomTouchListener customTouchListener) {
        mCustomTouchListener = customTouchListener;
    }

    @Override
    public boolean onTouch(View view, MotionEvent motionEvent) {
        mScaleGestureDetector.onTouchEvent(motionEvent);
        if (!mScaleGestureDetector.isInProgress()) {
            mGestureDetector.onTouchEvent(motionEvent);
        }
        return true;
    }

    public interface CustomTouchListener {
        // 放大缩小
        void zoom(float delta);

        // 点击
        void click(float x, float y);

        // 双击
        void doubleClick(float x, float y);

        // 长按
        void longPress(float x, float y);
    }
}

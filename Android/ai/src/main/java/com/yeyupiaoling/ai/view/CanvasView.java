package com.yeyupiaoling.ai.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import com.yeyupiaoling.ai.bean.RecognitionResult;

import java.util.Arrays;
import java.util.List;

public class CanvasView extends View {
    private static final String TAG = CanvasView.class.getSimpleName();

    private final Paint mPaint = new Paint();
    private final Paint mTextPaint = new Paint();
    private int width;
    private int height;
    private List<RecognitionResult> mList;

    public CanvasView(final Context context, final AttributeSet attrs) {
        super(context, attrs);
        // 文字的画笔
        mTextPaint.setColor(Color.YELLOW);
        mTextPaint.setStyle(Paint.Style.FILL);
        mTextPaint.setAntiAlias(false);
        mTextPaint.setTextSize(30);
        mTextPaint.setFakeBoldText(true);
        // 框的画笔
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setColor(Color.CYAN);
        mPaint.setStrokeWidth(5.0f);
    }

    public void setSize(int width, int height){
        this.width = width;
        this.height = height;
    }

    // 获取预测数据并进行绘画
    public void populateResultList(List<RecognitionResult> list) {
        mList = list;
        postInvalidate();
    }

    @Override
    public void draw(Canvas canvas) {
        float widthRatio = (float) getWidth() / (float) this.width;
        float heightRatio = (float) getHeight() / (float) this.height;

        if (mList == null || mList.size() == 0) {
            canvas.drawColor(Color.TRANSPARENT);
            return;
        }
        for (RecognitionResult resultData : mList) {
            canvas.drawRoundRect(resultData.left * widthRatio, resultData.top * heightRatio,
                    resultData.right * widthRatio, resultData.bottom * heightRatio, 20, 20, mPaint);
            canvas.drawText(resultData.label, (resultData.left + 10) * widthRatio, (resultData.top - 10) * heightRatio, mTextPaint);
        }
        super.draw(canvas);
    }
}

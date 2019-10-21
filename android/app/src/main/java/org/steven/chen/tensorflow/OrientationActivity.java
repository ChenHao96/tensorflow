package org.steven.chen.tensorflow;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.view.OrientationEventListener;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class OrientationActivity extends AppCompatActivity {

    private OrientationEventListener mOrEventListener;

    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        super.setContentView(R.layout.activity_orientation);
        super.setTitle(R.string.app_orientation_activity);
        this.textView = findViewById(R.id.text_orientation);
        this.startOrientationChangeListener();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (this.mOrEventListener != null) {
            this.mOrEventListener.disable();
            this.mOrEventListener = null;
        }
    }

    private void startOrientationChangeListener() {
        if (this.mOrEventListener != null) this.mOrEventListener.disable();
        this.mOrEventListener = new OrientationEventListener(this) {
            @Override
            @SuppressLint("DefaultLocale")
            public void onOrientationChanged(int rotation) {
                OrientationActivity.this.textView.setText(String.format("当前屏幕手持角度方法:%d°", rotation));
            }
        };
        this.mOrEventListener.enable();
    }
}

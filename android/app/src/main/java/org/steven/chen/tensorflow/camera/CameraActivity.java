package org.steven.chen.tensorflow.camera;

import android.graphics.Point;
import android.hardware.Camera;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.FrameLayout;

import androidx.appcompat.app.AppCompatActivity;

import org.steven.chen.tensorflow.R;

import java.util.Objects;

public class CameraActivity extends AppCompatActivity {

    private int touchCount = 0;
    private long lastTouchTime = 0L;
    private static final int QUICK_TOUCH_TIME = 140;

    private FrameLayout preview;
    private UserFocusPreview focusPreview;

    private Camera mCamera;
    private int cameraFacing = 0;
    private SurfaceView cameraPreview;

    private static final int[] cameraFacings = new int[]{
            Camera.CameraInfo.CAMERA_FACING_BACK,
            Camera.CameraInfo.CAMERA_FACING_FRONT
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        super.setContentView(R.layout.activity_camera);

        Objects.requireNonNull(getSupportActionBar()).hide();
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        this.preview = findViewById(R.id.camera_preview);
        this.focusPreview = new UserFocusPreview(this);
        this.preview.addView(this.focusPreview);
        this.switchCamera(cameraFacings[this.cameraFacing]);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        CameraUtil.releaseCamera(this.mCamera);
        this.mCamera = null;
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        if (MotionEvent.ACTION_DOWN == event.getAction()) {
            if (System.currentTimeMillis() - this.lastTouchTime > QUICK_TOUCH_TIME) {
                this.touchCount = 0;
            }
            this.touchProcess(++this.touchCount, event);
            this.lastTouchTime = System.currentTimeMillis();
        }
        return super.dispatchTouchEvent(event);
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        switch (keyCode) {
            case KeyEvent.KEYCODE_VOLUME_DOWN:
                CameraUtil.updateCameraZoom(this.mCamera, -5);
                break;
            case KeyEvent.KEYCODE_VOLUME_UP:
                CameraUtil.updateCameraZoom(this.mCamera, 5);
                break;
        }
        return super.onKeyDown(keyCode, event);
    }

    private void touchProcess(int touchCount, MotionEvent event) {
        if (touchCount == 2) {
            //switch camera
            this.cameraFacing++;
            this.cameraFacing %= cameraFacings.length;
            this.switchCamera(cameraFacings[this.cameraFacing]);
        } else if (touchCount == 1) {
            //camera focus
            Point point = new Point();
            point.x = (int) event.getX();
            point.y = (int) event.getY();
            int focusResult = CameraUtil.userClickFocus(this.cameraPreview, this.mCamera, point);
            if (focusResult >= CameraUtil.SUCCESS_CODE) {
                this.focusPreview.setUserFocusArea(point);
            }
        }
    }

    private void switchCamera(int cameraId) {
        CameraUtil.releaseCamera(this.mCamera);
        Camera camera = CameraUtil.getCameraInstance(this, cameraId);
        if (camera != null) {
            if (this.cameraPreview != null) this.preview.removeView(this.cameraPreview);
            this.cameraPreview = new CameraPreview(this, camera);
            this.preview.addView(this.cameraPreview);
            this.focusPreview.clearFocusArea();
            this.mCamera = camera;
            this.mCamera.setPreviewCallbackWithBuffer(new CameraPreviewBufferCallback(focusPreview));
        }
    }
}

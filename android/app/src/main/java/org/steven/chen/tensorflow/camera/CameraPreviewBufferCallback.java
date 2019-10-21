package org.steven.chen.tensorflow.camera;

import android.hardware.Camera;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraPreviewBufferCallback implements Camera.PreviewCallback {

    private byte[] copyData;
    private volatile boolean lock;
    private UserFocusPreview focusPreview;
//    private ExecutorService threadPool = Executors.newSingleThreadExecutor();

    public CameraPreviewBufferCallback(UserFocusPreview focusPreview) {
        this.focusPreview = focusPreview;
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {
        if (!this.lock) {
            if (this.copyData == null) {
                this.copyData = new byte[data.length];
            }
            System.arraycopy(data, 0, this.copyData, 0, this.copyData.length);
            this.lock = true;
//            threadPool.submit(new Runnable() {
//                @Override
//                public void run() {
//                    //TODO:
//                    lock = false;
//                }
//            });
        }
        camera.addCallbackBuffer(data);
    }
}
